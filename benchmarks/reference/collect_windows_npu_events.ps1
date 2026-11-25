param(
    [Parameter(Mandatory = $true)][DateTimeOffset]$Start,
    [DateTimeOffset]$End = [DateTimeOffset]::Now
)

$ErrorActionPreference = 'Stop'
if ($End -le $Start) { throw 'End must be after Start' }

function Read-SystemEvents([int[]]$Ids) {
    $query = @{LogName = 'System'; Id = $Ids; StartTime = $Start.LocalDateTime; EndTime = $End.LocalDateTime}
    try {
        Get-WinEvent -FilterHashtable $query -ErrorAction Stop
    } catch {
        if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound,*') { throw }
    }
}

$healthPattern = '\APID=(?<pid>[0-9]+)\s*Ctx=(?<context>[0-9]+)\s*TxnOp=(?<transaction>[0-9a-f]+)\s*CtxPC=(?<contextPc>[0-9a-f]+)\s*FeTYPE=(?<fatalType>[0-9a-f]+)\s*FeExTYPE=(?<exceptionType>[0-9a-f]+)\s*FePC=(?<exceptionPc>[0-9a-f]+)\s*FeAM=(?<module>[0-9a-f]+)\s*\z'
$contexts = @(Read-SystemEvents 3 | Where-Object ProviderName -eq 'pci' | Sort-Object RecordId | ForEach-Object {
    $result = [ordered]@{time = $_.TimeCreated.ToString('o'); record = $_.RecordId; parsed = $false}
    # The Windows NPU driver supplies its health record as an insertion string.
    # Message translation and even separators depend on the Windows driver package.
    foreach ($property in $_.Properties) {
        if ($property.Value -is [string] -and $property.Value -match $healthPattern) {
            $result.parsed = $true
            $result.pid = [int]$Matches.pid
            $result.context = [int]$Matches.context
            foreach ($field in @('transaction', 'contextPc', 'fatalType', 'exceptionType', 'exceptionPc', 'module')) {
                $result[$field] = $Matches[$field].ToLowerInvariant()
            }
            break
        }
    }
    [pscustomobject]$result
})
$power = @(Read-SystemEvents @(42, 107, 506, 507) | Where-Object ProviderName -eq 'Microsoft-Windows-Kernel-Power' |
    Sort-Object RecordId | ForEach-Object {
        @{time = $_.TimeCreated.ToString('o'); record = $_.RecordId; id = $_.Id}
    })

@{start = $Start.ToString('o'); end = $End.ToString('o'); pci_context_errors = $contexts; power_events = $power} |
    ConvertTo-Json -Depth 6
