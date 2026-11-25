"""Power and PCI evidence for the Windows profiler run."""

from contextlib import contextmanager
import ctypes as ct
from ctypes import wintypes as win
import json
from pathlib import Path
import subprocess


class DetailedReason(ct.Structure):
    _fields_ = [
        ('module', win.HMODULE), ('id', win.WORD), ('count', win.DWORD),
        ('strings', ct.POINTER(win.LPWSTR)),
    ]


class Reason(ct.Union):
    _fields_ = [('simple', win.LPWSTR), ('detailed', DetailedReason)]


class ReasonContext(ct.Structure):
    _fields_ = [('version', win.ULONG), ('flags', win.DWORD), ('reason', Reason)]


class SystemPowerStatus(ct.Structure):
    _fields_ = [
        ('ac_line', win.BYTE), ('battery_flags', win.BYTE), ('battery_percent', win.BYTE),
        ('battery_saver', win.BYTE), ('battery_seconds', win.DWORD), ('full_battery_seconds', win.DWORD),
    ]


def power_status(kernel):
    status = SystemPowerStatus()
    if not kernel.GetSystemPowerStatus(ct.byref(status)):
        return dict(read_error=ct.get_last_error())
    return dict(source={0: 'battery', 1: 'ac'}.get(status.ac_line, 'unknown'),
                battery_saver=bool(status.battery_saver))


@contextmanager
def keep_awake(keep_display=False):
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.PowerCreateRequest.argtypes = [ct.POINTER(ReasonContext)]
    kernel.PowerCreateRequest.restype = win.HANDLE
    for method in (kernel.PowerSetRequest, kernel.PowerClearRequest):
        method.argtypes = [win.HANDLE, ct.c_int]
        method.restype = win.BOOL
    kernel.GetSystemPowerStatus.argtypes = [ct.POINTER(SystemPowerStatus)]
    kernel.GetSystemPowerStatus.restype = win.BOOL
    kernel.CloseHandle.argtypes = [win.HANDLE]
    kernel.CloseHandle.restype = win.BOOL
    reason = ReasonContext(0, 1, Reason(simple='Phlegm Inkling prefill profiler'))
    handle = kernel.PowerCreateRequest(ct.byref(reason))
    if handle in (None, ct.c_void_p(-1).value):
        raise ct.WinError(ct.get_last_error())
    display_required, system_required, execution_required = 0, 1, 3
    active = []
    try:
        if not kernel.PowerSetRequest(handle, system_required):
            raise ct.WinError(ct.get_last_error())
        active.append(system_required)
        requests = ['system']
        report = dict(requests=requests)
        if kernel.PowerSetRequest(handle, execution_required):
            active.append(execution_required)
            requests.append('execution')
        else:
            report['execution_request_error'] = ct.get_last_error()
        if keep_display:
            if not kernel.PowerSetRequest(handle, display_required):
                raise ct.WinError(ct.get_last_error())
            active.append(display_required)
            requests.append('display')
        report['start'] = power_status(kernel)
        try:
            yield report
        finally:
            report['end'] = power_status(kernel)
    finally:
        for request in reversed(active):
            kernel.PowerClearRequest(handle, request)
        kernel.CloseHandle(handle)


def collect_events(start, end, path):
    collector = Path(__file__).resolve().parents[1] / 'collect_windows_npu_events.ps1'
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-File', collector, '-Start', start.isoformat(), '-End', end.isoformat()],
        check=True, capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    path.write_bytes(result.stdout)
    events = json.loads(result.stdout)
    return dict(file=path.name, power_events=len(events['power_events']),
                pci_context_errors=len(events['pci_context_errors']))
