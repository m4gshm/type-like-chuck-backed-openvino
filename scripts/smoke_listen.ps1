# Two-client smoke test for --listen mode (run from the repo root).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$ov = Join-Path $root '.cache\ov-genai\windows\openvino_genai_windows_2026.4.1.0_x86_64\runtime'
$env:PATH = (Join-Path $ov 'bin\intel64\Release') + ';' + (Join-Path $ov '3rdparty\tbb\bin') + ';' + $env:PATH
$runner = Join-Path $root 'build\tlc-ovgenai-runner-win32-x64.exe'
# Int4-sym (awq, g128) compiles fast on the NPU and runs on CPU — the
# int4/nf4-ov variants are slow to compile there (see e2e.py SUITE notes).
$model = Join-Path $root '.cache\models\qwen2.5-coder-0.5b-int4-sym'
$pipe = '\\.\pipe\tlc-runner-smoke-test'

function Connect-Client {
    $p = New-Object System.IO.Pipes.NamedPipeClientStream('.', 'tlc-runner-smoke-test', [System.IO.Pipes.PipeDirection]::InOut)
    $p.Connect(30000)
    return $p
}

function Read-Line($p) {
    $reader = [System.IO.StreamReader]::new($p, [System.Text.Encoding]::UTF8, $false, 4096, $true)
    return $reader.ReadLine()
}

function Send($p, $obj) {
    $writer = [System.IO.StreamWriter]::new($p, [System.Text.Encoding]::UTF8, 4096, $true)
    $writer.WriteLine(($obj | ConvertTo-Json -Compress -Depth 5))
    $writer.Flush()
}

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $runner
$psi.Arguments = "--model `"$model`" --device CPU --listen `"$pipe`" --token s3cret"
$psi.RedirectStandardError = $true
$psi.RedirectStandardOutput = $true
$psi.UseShellExecute = $false
$proc = [System.Diagnostics.Process]::Start($psi)

Start-Sleep -Seconds 20  # model load

function Send-Auth($p) { Send $p @{ type = 'auth'; token = 's3cret' } }

$c1 = Connect-Client
Send-Auth $c1
$ready1 = Read-Line $c1
Write-Host "client1 ready: $ready1"

# taken pipe name: second runner on the same endpoint must fail with exit 1
$psi2 = [System.Diagnostics.ProcessStartInfo]::new()
$psi2.FileName = $runner
$psi2.Arguments = "--model `"$model`" --device CPU --listen `"$pipe`""
$psi2.RedirectStandardError = $true
$psi2.UseShellExecute = $false
$proc2 = [System.Diagnostics.Process]::Start($psi2)
$proc2.WaitForExit(30000) | Out-Null
Write-Host "second runner on taken endpoint: exit=$($proc2.ExitCode)"

$c2 = Connect-Client
Send-Auth $c2
$ready2 = Read-Line $c2
Write-Host "client2 ready: $ready2"

# a token-less client must be rejected: error event, then the server closes
$bad = Connect-Client
[void] (Read-Line $bad)   # the ready broadcast arrives before auth
Send $bad @{ type = 'generate'; id = 1; prompt = 'x'; maxTokens = 4; stop = @() }
$err = (Read-Line $bad) | ConvertFrom-Json
if ($err.type -ne 'error') { Write-Host "FAIL: expected auth error, got $err"; exit 1 }
Write-Host "token-less client rejected"
$bad.Dispose()

Send $c1 @{ type = 'generate'; id = 7; prompt = 'def add(a, b):'; maxTokens = 16; stop = @("\n") }
$toks = @()
while ($true) {
    $ev = (Read-Line $c1) | ConvertFrom-Json
    if ($ev.type -eq 'token') { $toks += $ev.text; if ($ev.id -ne 7) { Write-Host "WRONG TOKEN ID: $($ev.id)"; exit 1 } }
    elseif ($ev.type -eq 'done') { Write-Host "client1 done: finish=$($ev.finishReason) tokens=$($toks.Count) id=$($ev.id)"; break }
    elseif ($ev.type -eq 'error') { Write-Host "client1 error: $($ev.message)"; exit 1 }
}

Send $c2 @{ type = 'generate'; id = 7; prompt = 'def mul(a, b):'; maxTokens = 16; stop = @("\n") }
while ($true) {
    $ev = (Read-Line $c2) | ConvertFrom-Json
    if ($ev.type -eq 'token' -and $ev.id -ne 7) { Write-Host "WRONG TOKEN ID: $($ev.id)"; exit 1 }
    elseif ($ev.type -eq 'done') { Write-Host "client2 done: finish=$($ev.finishReason) id=$($ev.id)"; break }
    elseif ($ev.type -eq 'error') { Write-Host "client2 error: $($ev.message)"; exit 1 }
}

# client1 leaves: the runner must keep serving client2
$c1.Dispose()
Start-Sleep -Seconds 2
if ($proc.HasExited) { Write-Host "FAIL: runner exited after first client left"; exit 1 }
Write-Host "runner alive after client1 left"

Send $c2 @{ type = 'generate'; id = 8; prompt = 'def sub(a, b):'; maxTokens = 16; stop = @("\n") }
while ($true) {
    $ev = (Read-Line $c2) | ConvertFrom-Json
    if ($ev.type -eq 'done') { Write-Host "client2 served after client1 left (id=$($ev.id))"; break }
    elseif ($ev.type -eq 'error') { Write-Host "client2 error: $($ev.message)"; exit 1 }
}

# last client leaves: the runner must self-exit
$c2.Dispose()
$exited = $proc.WaitForExit(15000)
Write-Host "runner self-exit after last client: $exited (exit=$($proc.ExitCode))"
if (-not $exited) { $proc.Kill(); exit 1 }
if ($proc.ExitCode -ne 0) { Write-Host "FAIL: nonzero exit $($proc.ExitCode)"; exit 1 }

# a client that dies mid-generation (writes fail) is dropped via the broken
# sink path, and with no other client the runner self-exits
$proc = [System.Diagnostics.Process]::Start($psi)
Start-Sleep -Seconds 15
$c3 = Connect-Client
[void] (Read-Line $c3)
Send $c3 @{ type = 'generate'; id = 1; prompt = 'tell me a very long story'; maxTokens = 8192; stop = @() }
Start-Sleep -Seconds 3   # generation is streaming tokens into the pipe
$c3.Dispose()            # die without reading: writes to it must fail
$exited = $proc.WaitForExit(30000)
Write-Host "dead client dropped, runner exited: $exited (exit=$($proc.ExitCode))"
if (-not $exited) { $proc.Kill(); Write-Host 'FAIL: dead client not dropped'; exit 1 }
if ($proc.ExitCode -ne 0) { Write-Host "FAIL: nonzero exit $($proc.ExitCode)"; exit 1 }

# TCP transport: spawner learns the random port from the child's stdout ready
$psiTcp = [System.Diagnostics.ProcessStartInfo]::new()
$psiTcp.FileName = $runner
$psiTcp.Arguments = "--model `"$model`" --device CPU --listen-tcp 0 --token s3cret"
$psiTcp.RedirectStandardError = $true
$psiTcp.RedirectStandardOutput = $true
$psiTcp.UseShellExecute = $false
$procTcp = [System.Diagnostics.Process]::Start($psiTcp)
$tcpReady = ($procTcp.StandardOutput.ReadLine()) | ConvertFrom-Json
if ($tcpReady.type -ne 'ready' -or -not $tcpReady.port) { Write-Host "FAIL: tcp ready event: $tcpReady"; exit 1 }
Write-Host "tcp ready: port=$($tcpReady.port) pid=$($tcpReady.pid)"

$tc = [System.Net.Sockets.TcpClient]::new('127.0.0.1', [int]$tcpReady.port)
$ts = $tc.GetStream()
$tw = [System.IO.StreamWriter]::new($ts, [System.Text.Encoding]::UTF8, 4096, $true)
$tr = [System.IO.StreamReader]::new($ts, [System.Text.Encoding]::UTF8, $false, 4096, $true)
$tw.WriteLine('{"type":"auth","token":"s3cret"}'); $tw.Flush()
$tw.WriteLine('{"type":"generate","id":5,"prompt":"def div(a, b):","maxTokens":16,"stop":["\n"]}'); $tw.Flush()
while ($true) {
    $ev = $tr.ReadLine() | ConvertFrom-Json
    if ($ev.type -eq 'token' -and $ev.id -ne 5) { Write-Host "FAIL: wrong token id $ev"; exit 1 }
    if ($ev.type -eq 'done') { Write-Host "tcp done: finish=$($ev.finishReason) id=$($ev.id)"; break }
    if ($ev.type -eq 'error') { Write-Host "FAIL: tcp error $ev"; exit 1 }
}
$tc.Close()
$exited = $procTcp.WaitForExit(15000)
if (-not ($exited -and $procTcp.ExitCode -eq 0)) { Write-Host "FAIL: tcp runner exit $($procTcp.ExitCode)"; exit 1 }
Write-Host "tcp runner self-exit: exit=0"

Write-Host 'SMOKE OK'
