package agentnames // import "github.com/agentstation/agentnames"

import (
	"strings"
	"testing"
)

// TestNameFormat tests the format of the generated name.
func TestNameFormat(t *testing.T) {
	name := Generate(0)
	if !strings.Contains(name, "_") {
		t.Fatalf("Generated name does not contain an underscore")
	}
	if strings.ContainsAny(name, "0123456789") {
		t.Fatalf("Generated name contains numbers!")
	}
}

// TestNameRetries tests the retries of the generated name.
func TestNameRetries(t *testing.T) {
	name := Generate(1)
	if !strings.Contains(name, "_") {
		t.Fatalf("Generated name does not contain an underscore")
	}
	if !strings.ContainsAny(name, "0123456789") {
		t.Fatalf("Generated name doesn't contain a number")
	}
}

// BenchmarkGetRandomName benchmarks the generation of a random name.
func BenchmarkGetRandomName(b *testing.B) {
	b.ReportAllocs()
	var out string
	for n := 0; n < b.N; n++ {
		out = Generate(5)
	}
	b.Log("Last result:", out)
}
