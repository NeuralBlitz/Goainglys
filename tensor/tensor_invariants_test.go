package tensor

import "testing"

func TestReshapePreservesGradientRequirementAndStorage(t *testing.T) {
	original := New(2, 3)
	original.ReqGrad = true
	original.Data[0] = 42
	original.Grad[0] = 7
	shape := []int{3, 2}

	reshaped := original.Reshape(shape...)
	if !reshaped.ReqGrad {
		t.Fatal("Reshape dropped ReqGrad")
	}
	if reshaped.Get(0, 0) != 42 || reshaped.Grad[0] != 7 {
		t.Fatal("Reshape did not preserve underlying data and gradient storage")
	}
	reshaped.Set(99, 0, 0)
	if original.Get(0, 0) != 99 {
		t.Fatal("Reshape should share underlying tensor storage")
	}
	shape[0] = 1
	if reshaped.Shape[0] != 3 {
		t.Fatal("Reshape retained caller-owned shape slice")
	}
}

func TestIndexRejectsOutOfBoundsCoordinates(t *testing.T) {
	x := New(2, 3)
	testCases := []struct {
		name string
		indices []int
	}{
		{"row too large but flat index valid", []int{2, 0}},
		{"negative row", []int{-1, 2}},
		{"column too large", []int{0, 3}},
		{"negative column", []int{1, -1}},
		{"wrong rank", []int{1}},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("Index%v should panic", tc.indices)
				}
			}()
			x.Index(tc.indices...)
		})
	}
	if got := x.Index(1, 2); got != 5 {
		t.Fatalf("Index(1,2) = %d, want 5", got)
	}
}
