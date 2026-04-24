package symbols

// StandaloneFunc is a top-level function.
func StandaloneFunc(x int) int {
	return x + 1
}

// Server is a struct type.
type Server struct {
	timeout int
}

// Start is a method on Server.
func (s *Server) Start() error {
	return nil
}

// Handler is an interface.
type Handler interface {
	// ServeHTTP is an interface method stub — must NOT be matched by find_symbol_range("ServeHTTP").
	ServeHTTP(w int, r int)
}

// Option is a type alias for a functional option.
type Option func(*Server)

// WithTimeout returns a functional Option.
func WithTimeout(timeout int) Option {
	return func(s *Server) {
		s.timeout = timeout
	}
}
