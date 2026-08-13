param([Parameter(Mandatory=$true)][string]$HostName,[Parameter(Mandatory=$true)][string]$RemoteArchive,[string]$UserName="root",[int]$Port=22,[string]$Destination=".")
scp -P $Port -- "$UserName@$HostName`:$RemoteArchive" $Destination
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Destination (Split-Path -Leaf $RemoteArchive))
