package loop

import (
	"context"

	"gitcode.com/mindspore/mscli/internal/pathpolicy"
)

type PathAuthorizationScope string

const (
	PathAuthorizationDeny       PathAuthorizationScope = "deny"
	PathAuthorizationOnce       PathAuthorizationScope = "once"
	PathAuthorizationSession    PathAuthorizationScope = "session"
	PathAuthorizationPersistent PathAuthorizationScope = "persistent"
)

type PathAuthorizationDecision struct {
	Scope PathAuthorizationScope
	Root  string
	Mode  string
}

const (
	PathAuthorizationModeRead  = "read"
	PathAuthorizationModeWrite = "write"
)

type PathAuthorizer interface {
	RequestPathAuthorization(ctx context.Context, denial *pathpolicy.PathDenial) (PathAuthorizationDecision, error)
}
