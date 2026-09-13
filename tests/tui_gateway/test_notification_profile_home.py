"""Real-thread regression for #78497; all ledgers live under tmp_path."""
import json
import queue
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
from tools import async_delegation as ad
from tools.process_registry import process_registry
from tui_gateway import server


def _row(home, delegation_id):
    with sqlite3.connect(home / "state.db") as db:
        return db.execute(
            "SELECT delivery_state, delivery_attempts, delivered_at FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


@pytest.mark.parametrize("profile", ["default", "nova", "byte"])
@pytest.mark.parametrize("drain", [False, True])
def test_poller_claims_and_acks_session_home(monkeypatch, tmp_path, profile, drain):
    launch = tmp_path / "launch"
    launch.mkdir()
    home = launch if profile == "default" else tmp_path / profile
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    for root in {launch, home}:
        with sqlite3.connect(root / "state.db") as db:
            ad._initialize_schema(db)
    event = dict(type="async_delegation", delegation_id="deleg_profile_test",
                 session_key="session-test", origin_ui_session_id="sid-test",
                 status="completed", summary="done")
    with sqlite3.connect(home / "state.db") as db:
        db.execute("""INSERT INTO async_delegations
            (delegation_id, origin_session, state, dispatched_at, updated_at, event_json)
            VALUES (?, ?, 'completed', ?, ?, ?)""",
            (event["delegation_id"], event["session_key"], time.time(), time.time(), json.dumps(event)))
    session = dict(profile_home=str(home), session_key="session-test", running=False,
                   history_lock=threading.Lock(), agent=SimpleNamespace())
    monkeypatch.setattr(server, "_sessions", {"sid-test": session})
    events = queue.Queue()
    events.put(event)
    monkeypatch.setattr(process_registry, "completion_queue", events)
    monkeypatch.setattr(server, "_wire_desktop_sinks", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    for name in ("_poll_bot_live_delivery_once", "_maybe_fire_tui_loop_tick",
                 "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban"):
        monkeypatch.setattr(server, name, lambda *a: None)
    accepted = threading.Event()
    turns = []
    def submit(*args, **kwargs):
        turns.append((get_hermes_home(), kwargs))
        accepted.set()
        return True
    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    if drain:
        # A pre-set stop drives the real shutdown drain rather than its idle loop.
        original_loop = server._notification_poller_loop
        def stopped_loop(stop, sid, state):
            stop.set()
            original_loop(stop, sid, state)
        monkeypatch.setattr(server, "_notification_poller_loop", stopped_loop)
    # Deliberately wrong caller override: session metadata, not caller context,
    # is authoritative (compute-host init can run after resetting its override).
    token = set_hermes_home_override(tmp_path / "unrelated-caller")
    try:
        stop = server._start_notification_poller("sid-test", session)
        worker = server._notification_pollers[-1][1]
        try:
            assert accepted.wait(5)
        finally:
            stop.set()
            worker.join(5)
        assert not worker.is_alive()
        assert get_hermes_home() == tmp_path / "unrelated-caller"
    finally:
        reset_hermes_home_override(token)
    row = _row(home, event["delegation_id"])
    assert row[0] == "delivered"
    assert row[1] == 1
    assert row[2] is not None
    assert len(turns) == 1
    assert turns[0][0] == home
    assert turns[0][1]["display_kind"] == "async_delegation_complete"
    if home != launch:
        assert _row(launch, event["delegation_id"]) is None
    token = set_hermes_home_override(home)
    try:
        assert ad.claim_event_delivery(event, "second-consumer") is None
    finally:
        reset_hermes_home_override(token)


def test_compute_host_supplies_home_before_session_starts(monkeypatch, tmp_path):
    from tui_gateway.compute_host import ComputeHost
    import hermes_state_registry

    host = object.__new__(ComputeHost)
    host._transport = SimpleNamespace()
    monkeypatch.setattr(hermes_state_registry, "acquire", lambda path: None)
    initialized = []
    sessions = {}
    def init(sid, key, agent, history, **kwargs):
        # _init_session spawns the poller before returning. Recording the
        # argument here catches late profile_home assignment deterministically.
        initialized.append(kwargs.get("profile_home"))
        sessions[sid] = {}
    fake_server = SimpleNamespace(
        _sessions=sessions, _make_agent=lambda *a, **kw: SimpleNamespace(),
        _transfer_db_to_agent=lambda *a: False, _init_session=init,
    )
    home = str(tmp_path / "named")
    host._ensure_server_session(fake_server, dict(sid="sid", profile_home=home))
    assert initialized == [home]


@pytest.mark.parametrize("raises", [False, True])
def test_poller_home_override_is_reset(monkeypatch, tmp_path, raises):
    original = tmp_path / "caller"
    home = tmp_path / "session"
    def loop(*args):
        assert get_hermes_home() == home
        if raises:
            raise RuntimeError("test poller failure")
    monkeypatch.setattr(server, "_notification_poller_loop", loop)
    token = set_hermes_home_override(original)
    try:
        if raises:
            with pytest.raises(RuntimeError, match="test poller failure"):
                server._run_notification_poller_in_session_home(threading.Event(), "sid", {"profile_home": home})
        else:
            server._run_notification_poller_in_session_home(threading.Event(), "sid", {"profile_home": home})
        assert get_hermes_home() == original
    finally:
        reset_hermes_home_override(token)
