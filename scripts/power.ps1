# Shared helpers for 1_SETUP_AND_RUN.ps1 and 2_AI_ENGINEER.ps1 (dot-source this file).
#
#   Enter-NoSleep / Exit-NoSleep : no system sleep, no display sleep, lid close does
#                                  nothing (admin), original lid setting restored after.
#   Disable-QuickEdit           : a mouse click in a console with QuickEdit on FREEZES
#                                  the running script until a key is pressed; turn it off.
#   Test-Admin                  : $true when running elevated.

if (-not ("SplitShield.Native" -as [type])) {
    Add-Type -Namespace SplitShield -Name Native -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern System.IntPtr GetStdHandle(int nStdHandle);
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern bool GetConsoleMode(System.IntPtr h, out uint mode);
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern bool SetConsoleMode(System.IntPtr h, uint mode);
'@
}

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Disable-QuickEdit {
    try {
        $h = [SplitShield.Native]::GetStdHandle(-10)   # STD_INPUT_HANDLE
        $mode = 0
        if ([SplitShield.Native]::GetConsoleMode($h, [ref]$mode)) {
            # clear ENABLE_QUICK_EDIT_MODE (0x40), keep ENABLE_EXTENDED_FLAGS (0x80)
            [void][SplitShield.Native]::SetConsoleMode($h, (($mode -band (-bnot 0x40)) -bor 0x80))
        }
    } catch { Write-Host "  (could not change console QuickEdit mode: $_)" }
}

function Get-LidAction([string]$kind) {
    try {
        $line = powercfg /q SCHEME_CURRENT SUB_BUTTONS LIDACTION 2>$null |
            Select-String "Current $kind Power Setting Index"
        if ($line) { return [Convert]::ToInt32(($line.ToString() -split ':')[-1].Trim(), 16) }
    } catch { }
    return $null
}

$script:SavedLid = $null

function Enter-NoSleep {
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_AWAYMODE_REQUIRED
    [void][SplitShield.Native]::SetThreadExecutionState([uint32]2147483715)
    if (Test-Admin) {
        $ac = Get-LidAction "AC"; $dc = Get-LidAction "DC"
        if ($null -ne $ac) {
            $script:SavedLid = @($ac, $dc)
            powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 | Out-Null
            powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 | Out-Null
            powercfg /setactive SCHEME_CURRENT | Out-Null
        }
    }
    Write-Host "== machine kept awake (sleep, display and lid) until the run finishes =="
}

function Exit-NoSleep {
    [void][SplitShield.Native]::SetThreadExecutionState([uint32]2147483648)  # ES_CONTINUOUS only
    if ($null -ne $script:SavedLid) {
        powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION $script:SavedLid[0] | Out-Null
        powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION $script:SavedLid[1] | Out-Null
        powercfg /setactive SCHEME_CURRENT | Out-Null
        $script:SavedLid = $null
    }
    Write-Host "== power settings restored =="
}
