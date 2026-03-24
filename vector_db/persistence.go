package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	opInsert byte = 1
	opUpdate byte = 2
	opDelete byte = 3
)

// PersistentStore manages WAL and snapshots for VectorDB
type PersistentStore struct {
	dir       string
	walPath   string
	snapDir   string
	walFile   *os.File
	walWriter *bufio.Writer
	mu        sync.Mutex
	seqNum    uint64
}

// NewPersistentStore creates a store backed by the given directory
func NewPersistentStore(dir string) (*PersistentStore, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create store dir: %w", err)
	}
	snapDir := filepath.Join(dir, "snapshots")
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		return nil, fmt.Errorf("create snapshots dir: %w", err)
	}
	return &PersistentStore{
		dir:     dir,
		walPath: filepath.Join(dir, "vectors.wal"),
		snapDir: snapDir,
	}, nil
}

// OpenWAL opens the WAL file for appending
func (ps *PersistentStore) OpenWAL() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	f, err := os.OpenFile(ps.walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("open WAL: %w", err)
	}
	ps.walFile = f
	ps.walWriter = bufio.NewWriterSize(f, 64*1024)
	return nil
}

// CloseWAL flushes and closes the WAL file
func (ps *PersistentStore) CloseWAL() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ps.walWriter != nil {
		if err := ps.walWriter.Flush(); err != nil {
			return err
		}
	}
	if ps.walFile != nil {
		return ps.walFile.Close()
	}
	return nil
}

// WALAppendInsert records an insert operation
func (ps *PersistentStore) WALAppendInsert(id string, vector []float64, metadata map[string]interface{}) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.seqNum++
	return ps.writeWALEntry(opInsert, ps.seqNum, id, vector, metadata)
}

// WALAppendUpdate records an update operation
func (ps *PersistentStore) WALAppendUpdate(id string, vector []float64, metadata map[string]interface{}) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.seqNum++
	return ps.writeWALEntry(opUpdate, ps.seqNum, id, vector, metadata)
}

// WALAppendDelete records a delete operation
func (ps *PersistentStore) WALAppendDelete(id string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.seqNum++
	return ps.writeWALEntry(opDelete, ps.seqNum, id, nil, nil)
}

// writeWALEntry writes a single WAL entry.
// Format:
// [1 byte op] [8 byte seqNum] [4 byte idLen] [id bytes]
// For insert/update:
//
//	[4 byte dim] [8*dim float64] [4 byte metaJSONLen] [metaJSON bytes]
//
// For delete: nothing more
func (ps *PersistentStore) writeWALEntry(op byte, seq uint64, id string, vector []float64, meta map[string]interface{}) error {
	w := ps.walWriter
	if w == nil {
		return fmt.Errorf("WAL not open")
	}

	// op
	if err := w.WriteByte(op); err != nil {
		return err
	}
	// seqNum
	if err := binary.Write(w, binary.LittleEndian, seq); err != nil {
		return err
	}
	// id
	idBytes := []byte(id)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(idBytes))); err != nil {
		return err
	}
	if _, err := w.Write(idBytes); err != nil {
		return err
	}

	if op == opInsert || op == opUpdate {
		// vector dimension
		dim := len(vector)
		if err := binary.Write(w, binary.LittleEndian, uint32(dim)); err != nil {
			return err
		}
		// vector data
		for _, v := range vector {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
		// metadata as simple key=value lines (avoids importing encoding/json)
		metaBytes := serializeMetadata(meta)
		if err := binary.Write(w, binary.LittleEndian, uint32(len(metaBytes))); err != nil {
			return err
		}
		if len(metaBytes) > 0 {
			if _, err := w.Write(metaBytes); err != nil {
				return err
			}
		}
	}

	return w.Flush()
}

// ReadWAL replays all WAL entries after the given sequence number
func (ps *PersistentStore) ReadWAL(afterSeq uint64) ([]WALEntry, error) {
	f, err := os.Open(ps.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var entries []WALEntry
	r := bufio.NewReaderSize(f, 64*1024)

	for {
		entry, err := readWALEntry(r)
		if err == io.EOF {
			break
		}
		if err != nil {
			return entries, fmt.Errorf("read WAL entry: %w", err)
		}
		if entry.SeqNum > afterSeq {
			entries = append(entries, entry)
		}
	}
	return entries, nil
}

// WALEntry represents a single WAL record
type WALEntry struct {
	Op       byte
	SeqNum   uint64
	ID       string
	Vector   []float64
	Metadata map[string]interface{}
}

func readWALEntry(r *bufio.Reader) (WALEntry, error) {
	var entry WALEntry

	op, err := r.ReadByte()
	if err != nil {
		return entry, err
	}
	entry.Op = op

	if err := binary.Read(r, binary.LittleEndian, &entry.SeqNum); err != nil {
		return entry, err
	}

	var idLen uint32
	if err := binary.Read(r, binary.LittleEndian, &idLen); err != nil {
		return entry, err
	}
	idBytes := make([]byte, idLen)
	if _, err := io.ReadFull(r, idBytes); err != nil {
		return entry, err
	}
	entry.ID = string(idBytes)

	if op == opInsert || op == opUpdate {
		var dim uint32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			return entry, err
		}
		entry.Vector = make([]float64, dim)
		for i := range entry.Vector {
			if err := binary.Read(r, binary.LittleEndian, &entry.Vector[i]); err != nil {
				return entry, err
			}
		}

		var metaLen uint32
		if err := binary.Read(r, binary.LittleEndian, &metaLen); err != nil {
			return entry, err
		}
		if metaLen > 0 {
			metaBytes := make([]byte, metaLen)
			if _, err := io.ReadFull(r, metaBytes); err != nil {
				return entry, err
			}
			entry.Metadata = deserializeMetadata(metaBytes)
		}
	}

	return entry, nil
}

// CompactWAL rewrites the WAL using only the latest state per ID
func (ps *PersistentStore) CompactWAL(db *VectorDB) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Close current WAL
	if ps.walWriter != nil {
		if err := ps.walWriter.Flush(); err != nil {
			return err
		}
	}
	if ps.walFile != nil {
		if err := ps.walFile.Close(); err != nil {
			return err
		}
	}

	// Write new WAL from current state
	tmpPath := ps.walPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create compact WAL: %w", err)
	}

	w := bufio.NewWriterSize(f, 64*1024)
	seq := ps.seqNum

	// Write all current vectors as inserts
	for id, vec := range db.vectors {
		meta := db.metadata[id]
		seq++
		if err := writeEntryToWriter(w, opInsert, seq, id, vec, meta); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return err
		}
	}

	if err := w.Flush(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmpPath)
		return err
	}

	// Atomic rename
	if err := os.Rename(tmpPath, ps.walPath); err != nil {
		return fmt.Errorf("rename WAL: %w", err)
	}

	// Reopen
	f2, err := os.OpenFile(ps.walPath, os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	ps.walFile = f2
	ps.walWriter = bufio.NewWriterSize(f2, 64*1024)
	ps.seqNum = seq

	return nil
}

func writeEntryToWriter(w *bufio.Writer, op byte, seq uint64, id string, vector []float64, meta map[string]interface{}) error {
	w.WriteByte(op)
	binary.Write(w, binary.LittleEndian, seq)
	idBytes := []byte(id)
	binary.Write(w, binary.LittleEndian, uint32(len(idBytes)))
	w.Write(idBytes)

	if op == opInsert || op == opUpdate {
		binary.Write(w, binary.LittleEndian, uint32(len(vector)))
		for _, v := range vector {
			binary.Write(w, binary.LittleEndian, v)
		}
		metaBytes := serializeMetadata(meta)
		binary.Write(w, binary.LittleEndian, uint32(len(metaBytes)))
		if len(metaBytes) > 0 {
			w.Write(metaBytes)
		}
	}
	return nil
}

// SaveSnapshot writes a full snapshot of the database
func (ps *PersistentStore) SaveSnapshot(db *VectorDB) (string, error) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	timestamp := time.Now().Unix()
	snapFile := filepath.Join(ps.snapDir, fmt.Sprintf("snapshot_%d_%d.bin", timestamp, ps.seqNum))

	f, err := os.Create(snapFile)
	if err != nil {
		return "", fmt.Errorf("create snapshot: %w", err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 256*1024)

	// Header: magic + version + dimension + metric + seqNum + count
	magic := []byte("VDBS")
	w.Write(magic)
	binary.Write(w, binary.LittleEndian, uint32(1)) // version
	binary.Write(w, binary.LittleEndian, uint32(db.dimension))
	binary.Write(w, binary.LittleEndian, uint32(db.metric))
	binary.Write(w, binary.LittleEndian, ps.seqNum)
	binary.Write(w, binary.LittleEndian, uint32(len(db.vectors)))

	// Write each vector
	for id, vec := range db.vectors {
		idBytes := []byte(id)
		binary.Write(w, binary.LittleEndian, uint32(len(idBytes)))
		w.Write(idBytes)

		for _, v := range vec {
			binary.Write(w, binary.LittleEndian, v)
		}

		meta := db.metadata[id]
		metaBytes := serializeMetadata(meta)
		binary.Write(w, binary.LittleEndian, uint32(len(metaBytes)))
		if len(metaBytes) > 0 {
			w.Write(metaBytes)
		}
	}

	if err := w.Flush(); err != nil {
		return "", err
	}

	return snapFile, nil
}

// LoadSnapshot loads a snapshot file and populates the VectorDB
func (ps *PersistentStore) LoadSnapshot(db *VectorDB, path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, fmt.Errorf("open snapshot: %w", err)
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 256*1024)

	// Read header
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return 0, err
	}
	if string(magic) != "VDBS" {
		return 0, fmt.Errorf("invalid snapshot magic")
	}

	var version, dim, metric, count uint32
	var seqNum uint64
	binary.Read(r, binary.LittleEndian, &version)
	binary.Read(r, binary.LittleEndian, &dim)
	binary.Read(r, binary.LittleEndian, &metric)
	binary.Read(r, binary.LittleEndian, &seqNum)
	binary.Read(r, binary.LittleEndian, &count)

	if int(dim) != db.dimension {
		return 0, fmt.Errorf("dimension mismatch: snapshot=%d, db=%d", dim, db.dimension)
	}

	// Read vectors
	for i := uint32(0); i < count; i++ {
		var idLen uint32
		binary.Read(r, binary.LittleEndian, &idLen)
		idBytes := make([]byte, idLen)
		io.ReadFull(r, idBytes)
		id := string(idBytes)

		vec := make([]float64, dim)
		for j := range vec {
			binary.Read(r, binary.LittleEndian, &vec[j])
		}

		var metaLen uint32
		binary.Read(r, binary.LittleEndian, &metaLen)
		var meta map[string]interface{}
		if metaLen > 0 {
			metaBytes := make([]byte, metaLen)
			io.ReadFull(r, metaBytes)
			meta = deserializeMetadata(metaBytes)
		}

		db.vectors[id] = vec
		if meta != nil {
			db.metadata[id] = meta
		}
		db.index.Add(id, vec)
	}

	ps.seqNum = seqNum
	return seqNum, nil
}

// FindLatestSnapshot returns the path of the most recent snapshot
func (ps *PersistentStore) FindLatestSnapshot() (string, uint64, error) {
	entries, err := os.ReadDir(ps.snapDir)
	if err != nil {
		return "", 0, err
	}

	var bestPath string
	var bestSeq uint64

	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, "snapshot_") || !strings.HasSuffix(name, ".bin") {
			continue
		}
		// snapshot_{timestamp}_{seqnum}.bin
		trimmed := strings.TrimPrefix(name, "snapshot_")
		trimmed = strings.TrimSuffix(trimmed, ".bin")
		parts := strings.SplitN(trimmed, "_", 2)
		if len(parts) != 2 {
			continue
		}
		seq, err := strconv.ParseUint(parts[1], 10, 64)
		if err != nil {
			continue
		}
		if seq >= bestSeq {
			bestSeq = seq
			bestPath = filepath.Join(ps.snapDir, name)
		}
	}
	return bestPath, bestSeq, nil
}

// Recover restores the database from the latest snapshot + WAL replay
func Recover(db *VectorDB, store *PersistentStore) error {
	// Find latest snapshot
	snapPath, snapSeq, err := store.FindLatestSnapshot()
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("find snapshot: %w", err)
	}

	if snapPath != "" {
		if _, err := store.LoadSnapshot(db, snapPath); err != nil {
			return fmt.Errorf("load snapshot: %w", err)
		}
	}

	// Replay WAL entries after snapshot
	entries, err := store.ReadWAL(snapSeq)
	if err != nil {
		return fmt.Errorf("read WAL: %w", err)
	}

	for _, entry := range entries {
		switch entry.Op {
		case opInsert:
			db.vectors[entry.ID] = entry.Vector
			if entry.Metadata != nil {
				db.metadata[entry.ID] = entry.Metadata
			}
			db.index.Add(entry.ID, entry.Vector)
		case opUpdate:
			delete(db.vectors, entry.ID)
			delete(db.metadata, entry.ID)
			db.index.Delete(entry.ID)
			db.vectors[entry.ID] = entry.Vector
			if entry.Metadata != nil {
				db.metadata[entry.ID] = entry.Metadata
			}
			db.index.Add(entry.ID, entry.Vector)
		case opDelete:
			delete(db.vectors, entry.ID)
			delete(db.metadata, entry.ID)
			db.index.Delete(entry.ID)
		}
	}

	if len(entries) > 0 {
		store.seqNum = entries[len(entries)-1].SeqNum
	}

	return nil
}

// ListSnapshots returns all snapshot files sorted by sequence number
func (ps *PersistentStore) ListSnapshots() ([]SnapshotInfo, error) {
	entries, err := os.ReadDir(ps.snapDir)
	if err != nil {
		return nil, err
	}

	var infos []SnapshotInfo
	for _, e := range entries {
		if e.IsDir() || !strings.HasPrefix(e.Name(), "snapshot_") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		trimmed := strings.TrimPrefix(e.Name(), "snapshot_")
		trimmed = strings.TrimSuffix(trimmed, ".bin")
		parts := strings.SplitN(trimmed, "_", 2)
		if len(parts) != 2 {
			continue
		}
		ts, _ := strconv.ParseInt(parts[0], 10, 64)
		seq, _ := strconv.ParseUint(parts[1], 10, 64)
		infos = append(infos, SnapshotInfo{
			Path:      filepath.Join(ps.snapDir, e.Name()),
			Timestamp: time.Unix(ts, 0),
			SeqNum:    seq,
			Size:      info.Size(),
		})
	}

	sort.Slice(infos, func(i, j int) bool {
		return infos[i].SeqNum < infos[j].SeqNum
	})
	return infos, nil
}

// SnapshotInfo describes a snapshot file
type SnapshotInfo struct {
	Path      string
	Timestamp time.Time
	SeqNum    uint64
	Size      int64
}

// PurgeOldSnapshots keeps only the most recent N snapshots
func (ps *PersistentStore) PurgeOldSnapshots(keep int) error {
	snapshots, err := ps.ListSnapshots()
	if err != nil {
		return err
	}
	if len(snapshots) <= keep {
		return nil
	}
	for _, s := range snapshots[:len(snapshots)-keep] {
		os.Remove(s.Path)
	}
	return nil
}

// WALSize returns the current WAL file size in bytes
func (ps *PersistentStore) WALSize() int64 {
	info, err := os.Stat(ps.walPath)
	if err != nil {
		return 0
	}
	return info.Size()
}

// --- Metadata serialization (simple key:type:value format) ---

func serializeMetadata(meta map[string]interface{}) []byte {
	if meta == nil || len(meta) == 0 {
		return nil
	}
	var b []byte
	for k, v := range meta {
		keyBytes := []byte(k)
		valStr := fmt.Sprintf("%v", v)
		valBytes := []byte(valStr)
		// [2 keyLen][key][2 valLen][val]
		b = append(b, byte(len(keyBytes)>>8), byte(len(keyBytes)))
		b = append(b, keyBytes...)
		b = append(b, byte(len(valBytes)>>8), byte(len(valBytes)))
		b = append(b, valBytes...)
	}
	return b
}

func deserializeMetadata(data []byte) map[string]interface{} {
	if len(data) == 0 {
		return nil
	}
	meta := make(map[string]interface{})
	i := 0
	for i+4 <= len(data) {
		keyLen := int(data[i])<<8 | int(data[i+1])
		i += 2
		if i+keyLen > len(data) {
			break
		}
		key := string(data[i : i+keyLen])
		i += keyLen

		if i+2 > len(data) {
			break
		}
		valLen := int(data[i])<<8 | int(data[i+1])
		i += 2
		if i+valLen > len(data) {
			break
		}
		val := string(data[i : i+valLen])
		i += valLen

		// Try to parse as float64, otherwise keep as string
		if f, err := strconv.ParseFloat(val, 64); err == nil && !math.IsNaN(f) {
			meta[key] = f
		} else {
			meta[key] = val
		}
	}
	return meta
}
