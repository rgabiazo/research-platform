# Security Policy

## Supported alpha line

Security fixes for the planned public alpha will target the current `0.1.0`
alpha line, beginning with `0.1.0a1`. Earlier development snapshots and
superseded alpha builds will not receive separate fixes. This policy will be
reviewed as the project matures.

## Report a vulnerability privately

Do not open a public issue for a suspected, undisclosed vulnerability. Use
the private vulnerability reporting form:

https://github.com/rgabiazo/research-platform/security/advisories/new

Private vulnerability reporting is enabled for this repository, and the
maintainer monitors GitHub security-alert notifications.

A useful report includes:

- the affected component and version or commit;
- steps to reproduce the issue or a minimal proof of concept;
- the expected and observed behavior;
- the likely impact and any known preconditions; and
- suggested mitigations, if available.

Maintainers will acknowledge a report when it has been reviewed and will share
updates as investigation and remediation progress. Response or resolution
deadlines are not guaranteed during the alpha period.

Ordinary installation help, usage questions, scientific-method questions, and
non-sensitive defects are not vulnerability reports. After the public
repository is created, use its normal documentation and issue-tracking channels
for those topics.

## Repository hygiene

Never commit secrets, tokens, passwords, private keys, SSH material, private
study data, or sensitive logs. Keep credentials under the ignored `secrets/`
boundary and sanitize diagnostic output before sharing it.
