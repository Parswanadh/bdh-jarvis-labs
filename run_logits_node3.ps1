param(
    [switch]$Resume = $false,
    [switch]$GitPush = $false,
    [int]$BatchSize = 8,
    [int]$SeqLen = 192,
    [int]$TopK = 256,
    [string]$TeacherModel = "Qwen/Qwen2.5-0.5B-Instruct"
)

$argsList = @(
    "--batch-size", "$BatchSize",
    "--seq-len", "$SeqLen",
    "--top-k", "$TopK",
    "--teacher-model", "$TeacherModel"
)

if ($Resume) { $argsList += "--resume" }
if ($GitPush) { $argsList += "--git-push" }

python "generate_logits_node3.py" @argsList
exit $LASTEXITCODE

