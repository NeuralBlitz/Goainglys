package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

type ParameterServer struct {
	mu           sync.RWMutex
	parameters   map[string][]float32
	numWorkers   int
	readyWorkers int
}

func NewParameterServer() *ParameterServer {
	return &ParameterServer{
		parameters: make(map[string][]float32),
	}
}

func (ps *ParameterServer) RegisterWorker() int {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.numWorkers++
	return ps.numWorkers
}

func (ps *ParameterServer) InitializeParameter(name string, shape []int) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	size := 1
	for _, s := range shape {
		size *= s
	}

	ps.parameters[name] = make([]float32, size)
	for i := range ps.parameters[name] {
		ps.parameters[name][i] = float32(i) * 0.01
	}
}

func (ps *ParameterServer) GetParameter(name string) []float32 {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return ps.parameters[name]
}

func (ps *ParameterServer) GetAllParameters() map[string][]float32 {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	result := make(map[string][]float32)
	for k, v := range ps.parameters {
		cpy := make([]float32, len(v))
		copy(cpy, v)
		result[k] = cpy
	}
	return result
}

func (ps *ParameterServer) ApplyGradient(name string, grad []float32, lr float32) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	if len(ps.parameters[name]) != len(grad) {
		return
	}

	for i := range ps.parameters[name] {
		ps.parameters[name][i] -= lr * grad[i]
	}
}

type Worker struct {
	id           int
	server       *ParameterServer
	gradients    map[string][]float32
	learningRate float32
}

func NewWorker(id int, server *ParameterServer, lr float32) *Worker {
	return &Worker{
		id:           id,
		server:       server,
		gradients:    make(map[string][]float32),
		learningRate: lr,
	}
}

func (w *Worker) ComputeGradients(paramName string, param []float32) {
	size := len(param)
	grad := make([]float32, size)

	for i := range grad {
		grad[i] = param[i] * 0.1
	}

	w.gradients[paramName] = grad
}

func (w *Worker) PushGradients() {
	for name, grad := range w.gradients {
		w.server.ApplyGradient(name, grad, w.learningRate)
	}
}

func (w *Worker) PullParameters() map[string][]float32 {
	return w.server.GetAllParameters()
}

type DistributedConfig struct {
	NumWorkers    int
	BatchSize     int
	LearningRate  float32
	ServerAddress string
	WorkerAddress string
}

type DistributedTrainer struct {
	server     *ParameterServer
	workers    []*Worker
	config     DistributedConfig
	mu         sync.Mutex
	epoch      int
	totalSteps int
}

func NewDistributedTrainer(config DistributedConfig) *DistributedTrainer {
	server := NewParameterServer()
	workers := make([]*Worker, config.NumWorkers)

	for i := 0; i < config.NumWorkers; i++ {
		workers[i] = NewWorker(i, server, config.LearningRate)
	}

	return &DistributedTrainer{
		server:  server,
		workers: workers,
		config:  config,
	}
}

func (dt *DistributedTrainer) InitializeModel(paramShapes map[string][]int) {
	for name, shape := range paramShapes {
		dt.server.InitializeParameter(name, shape)
	}
}

func (dt *DistributedTrainer) TrainStep() map[string][]float32 {
	var wg sync.WaitGroup

	dt.mu.Lock()
	dt.totalSteps++
	dt.mu.Unlock()

	for _, worker := range dt.workers {
		wg.Add(1)
		go func(w *Worker) {
			defer wg.Done()

			params := w.PullParameters()
			for name, param := range params {
				w.ComputeGradients(name, param)
			}
			w.PushGradients()
		}(worker)
	}

	wg.Wait()

	return dt.server.GetAllParameters()
}

func (dt *DistributedTrainer) GetStats() map[string]interface{} {
	dt.mu.Lock()
	defer dt.mu.Unlock()

	return map[string]interface{}{
		"num_workers":   dt.config.NumWorkers,
		"epoch":         dt.epoch,
		"total_steps":   dt.totalSteps,
		"learning_rate": dt.config.LearningRate,
		"batch_size":    dt.config.BatchSize,
	}
}

type WorkerServer struct {
	worker          *Worker
	parameterServer string
	httpServer      *http.Server
}

func NewWorkerServer(worker *Worker, port int) *WorkerServer {
	ws := &WorkerServer{
		worker: worker,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/pull", ws.handlePull)
	mux.HandleFunc("/push", ws.handlePush)
	mux.HandleFunc("/stats", ws.handleStats)

	ws.httpServer = &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}

	return ws
}

func (ws *WorkerServer) handlePull(w http.ResponseWriter, r *http.Request) {
	params := ws.worker.PullParameters()
	json.NewEncoder(w).Encode(params)
}

func (ws *WorkerServer) handlePush(w http.ResponseWriter, r *http.Request) {
	var gradients map[string][]float32
	json.NewDecoder(r.Body).Decode(&gradients)

	for name, grad := range gradients {
		ws.worker.gradients[name] = grad
	}
	ws.worker.PushGradients()

	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (ws *WorkerServer) handleStats(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]int{"worker_id": ws.worker.id})
}

func (ws *WorkerServer) Start(ctx context.Context) error {
	return ws.httpServer.ListenAndServe()
}

func (ws *WorkerServer) Stop() error {
	return ws.httpServer.Shutdown(context.Background())
}

type AllReduce struct {
	mu      sync.Mutex
	context context.Context
	cancel  context.CancelFunc
}

func NewAllReduce() *AllReduce {
	ctx, cancel := context.WithCancel(context.Background())
	return &AllReduce{
		context: ctx,
		cancel:  cancel,
	}
}

func (ar *AllReduce) SumReduce(values []float32, worldSize int) float32 {
	var sum float32
	for _, v := range values {
		sum += v
	}
	return sum / float32(worldSize)
}

func (ar *AllReduce) Broadcast(value float32, root int, rank int) float32 {
	return value
}

type RingAllReduce struct {
	rank      int
	worldSize int
	mu        sync.Mutex
}

func NewRingAllReduce(rank, worldSize int) *RingAllReduce {
	return &RingAllReduce{
		rank:      rank,
		worldSize: worldSize,
	}
}

func (rar *RingAllReduce) AllReduceSum(data []float32) []float32 {
	result := make([]float32, len(data))
	copy(result, data)

	chunkSize := len(data) / rar.worldSize

	for i := 0; i < rar.worldSize-1; i++ {
		send := (rar.rank - i + rar.worldSize) % rar.worldSize
		recv := (rar.rank - i - 1 + rar.worldSize) % rar.worldSize

		_ = send
		_ = recv

		for j := 0; j < chunkSize; j++ {
			result[j] += data[j]
		}
	}

	return result
}

type GradientAggregation struct {
	mu           sync.Mutex
	gradients    map[string][]float32
	receivedFrom map[int]bool
	threshold    int
}

func NewGradientAggregation(numWorkers int) *GradientAggregation {
	return &GradientAggregation{
		gradients:    make(map[string][]float32),
		receivedFrom: make(map[int]bool),
		threshold:    numWorkers,
	}
}

func (ga *GradientAggregation) AddGradient(workerID int, grad map[string][]float32) bool {
	ga.mu.Lock()
	defer ga.mu.Unlock()

	ga.receivedFrom[workerID] = true

	for name, g := range grad {
		if ga.gradients[name] == nil {
			ga.gradients[name] = make([]float32, len(g))
			copy(ga.gradients[name], g)
		} else {
			for i := range ga.gradients[name] {
				ga.gradients[name][i] += g[i]
			}
		}
	}

	if len(ga.receivedFrom) >= ga.threshold {
		for name := range ga.gradients {
			for i := range ga.gradients[name] {
				ga.gradients[name][i] /= float32(ga.threshold)
			}
		}
		return true
	}

	return false
}

func (ga *GradientAggregation) GetAggregatedGradient() map[string][]float32 {
	ga.mu.Lock()
	defer ga.mu.Unlock()

	result := make(map[string][]float32)
	for k, v := range ga.gradients {
		cpy := make([]float32, len(v))
		copy(cpy, v)
		result[k] = cpy
	}

	ga.gradients = make(map[string][]float32)
	ga.receivedFrom = make(map[int]bool)

	return result
}

type SyncMode int

const (
	SyncAllReduce SyncMode = iota
	SyncParameterServer
	AsyncStochastic
)

type DistributedOptions struct {
	Mode          SyncMode
	NumWorkers    int
	Rank          int
	ServerAddress string
	LR            float32
}

type DistributedTrainingJob struct {
	config DistributedOptions
	server *ParameterServer
	jobID  string
	status string
	mu     sync.RWMutex
}

func NewDistributedTrainingJob(config DistributedOptions, jobID string) *DistributedTrainingJob {
	return &DistributedTrainingJob{
		config: config,
		server: NewParameterServer(),
		jobID:  jobID,
		status: "initialized",
	}
}

func (j *DistributedTrainingJob) Start() {
	j.mu.Lock()
	j.status = "running"
	j.mu.Unlock()
}

func (j *DistributedTrainingJob) Stop() {
	j.mu.Lock()
	j.status = "stopped"
	j.mu.Unlock()
}

func (j *DistributedTrainingJob) GetStatus() string {
	j.mu.RLock()
	defer j.mu.RUnlock()
	return j.status
}

func (j *DistributedTrainingJob) GetConfig() DistributedOptions {
	j.mu.RLock()
	defer j.mu.RUnlock()
	return j.config
}

type TrainingMetrics struct {
	Step         int
	Epoch        int
	Loss         float64
	LearningRate float32
	NumSamples   int
	Timestamp    time.Time
}

type MetricsCollector struct {
	mu      sync.RWMutex
	metrics []TrainingMetrics
	maxSize int
}

func NewMetricsCollector(maxSize int) *MetricsCollector {
	return &MetricsCollector{
		metrics: make([]TrainingMetrics, 0, maxSize),
		maxSize: maxSize,
	}
}

func (mc *MetricsCollector) Add(m TrainingMetrics) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	if len(mc.metrics) >= mc.maxSize {
		mc.metrics = mc.metrics[1:]
	}
	mc.metrics = append(mc.metrics, m)
}

func (mc *MetricsCollector) GetAll() []TrainingMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	result := make([]TrainingMetrics, len(mc.metrics))
	copy(result, mc.metrics)
	return result
}

func (mc *MetricsCollector) GetRecent(n int) []TrainingMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	if n > len(mc.metrics) {
		n = len(mc.metrics)
	}

	result := make([]TrainingMetrics, n)
	copy(result, mc.metrics[len(mc.metrics)-n:])
	return result
}

type Checkpoint struct {
	Step       int
	Epoch      int
	Parameters map[string][]float32
	Optimizer  map[string]interface{}
	Metrics    TrainingMetrics
}

func (c *Checkpoint) Save(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	_ = data
	return nil
}

func LoadCheckpoint(path string) (*Checkpoint, error) {
	var ckpt Checkpoint
	return &ckpt, nil
}
