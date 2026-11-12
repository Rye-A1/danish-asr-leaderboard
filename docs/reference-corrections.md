# Reference correction inbox

The public sample form sends suggestions to a private review spreadsheet. It
does not change benchmark references or scores. Review suggestions against the
audio before applying them in a separate change.

## Setup

1. Create a dedicated Google Cloud project and enable the Google Sheets, IAM,
   Security Token Service, and IAM Service Account Credentials APIs.
   Create a service account with **no project roles and no domain-wide
   delegation**. Confirm it has no inherited or other access to your resources.
2. Create a new private review spreadsheet with no attached scripts, add a tab
   named `Suggestions`, and put
   these headers in cells A1:H1: `Submitted (UTC)`, `Dataset`, `Clip ID`,
   `Original reference`, `Suggested reference`, `Sample link`, `Review status`,
   and `Submission ID`. Share only this spreadsheet with the service account as
   an editor. In the spreadsheet sharing settings, disable editors' ability to
   change permissions and share. Do not share the script project with it.
3. Create a **standalone**, private Apps Script project and copy
   `scripts/correction_inbox/Code.gs` and `appsscript.json` into it. The script
   requests `openid` and the Google external-request scope on the owner's
   account; it does not request access to the owner's Drive or Sheets. Keep the
   project private: its editors can change its code and use its permissions.
   Spreadsheet editors can edit scripts attached to that spreadsheet, even if
   editor sharing is disabled, so never attach this project to the review sheet.
4. Configure Workload Identity Federation with an OIDC provider whose issuer is
   `https://accounts.google.com`, whose allowed audience is the script's exact
   OAuth client ID, and whose attribute condition allows only the owner's exact
   subject. Map `google.subject=assertion.sub`. Run `identityClaimsForSetup` in
   the private editor to obtain the audience and subject, not the identity token.
   Grant `roles/iam.workloadIdentityUser` on the dedicated service account to
   that single pool subject, never the entire pool or a project-wide grant.
   Leave the service-account-key-creation policy enabled.
5. In the Apps Script project properties set `CORRECTIONS_SHEET_ID` to the
   spreadsheet ID, `SERVICE_ACCOUNT_EMAIL` to the dedicated service account,
   and `WIF_PROVIDER` to the provider's full resource name prefixed with
   `//iam.googleapis.com/`. Run `verifySetup` to check that the script can read
   the header of the restricted spreadsheet before continuing.
6. Deploy the script as a web app executing as the owner, with access for anyone,
   and put its `/exec` URL in `CORRECTION_ENDPOINT` in
   `huggingface/space/index.html`. The endpoint is public; the spreadsheet and
   script project must remain private. Submit a harmless test suggestion and
   check that one pending row appears and the form confirms it. Test a failure
   as well.

The script exchanges the owner's identity token for a short-lived token for the
service account. The federated principal has only the service-account
impersonation grant; the resulting Sheets token can read or modify any sheet the
service account can access. Check all its shares and inherited permissions.
The owner and other script editors can change the script, which can make network
requests. To revoke access, remove the spreadsheet share and the impersonation
grant or disable the service account and identity provider.

Submissions are untrusted. The endpoint validates IDs and lengths, writes cells
as raw values, deduplicates retries, and limits accepted submissions per day.
These protections do not prevent targeted spam, quota exhaustion, or disclosure
of information someone submits. Anyone who can edit the spreadsheet can read
its contents. The browser isolates the status response in a sandboxed iframe,
but a compromised endpoint could still falsely report submission status. Do not
use the inbox for private or sensitive recordings or text.
