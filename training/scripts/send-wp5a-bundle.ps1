param([Parameter(Mandatory=$true)][string]$HostName,[Parameter(Mandatory=$true)][string]$Archive,[string]$UserName="root",[int]$Port=22,[string]$RemotePath="/workspace")
$localHash=(Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLower()
scp -P $Port -- $Archive "$UserName@$HostName`:$RemotePath/"
Write-Host "Verify remotely: sha256sum $RemotePath/$(Split-Path -Leaf $Archive)  # expected $localHash"
