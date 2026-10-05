class_name EngineClient
extends Node
## Talks to the Python engine over a localhost socket (newline-delimited JSON).
## If no engine is listening it starts one: the packaged sidecar next to the
## app, or `python -m engine` from the repository when running from source.

signal connected
signal disconnected
signal event_received(msg: Dictionary)
signal engine_log(text: String)
## The engine we started is gone or never came up; `text` says why.
signal engine_failed(text: String)

const DEFAULT_PORT := 47321
const CONNECT_RETRY_SECONDS := 1.0
const SPAWN_AFTER_FAILED_TRIES := 2
const MAX_SPAWNS := 3
## The first start of a packaged engine unpacks and loads a lot (antivirus scans
## it too), so give it a while before calling it stuck.
const START_TIMEOUT_SECONDS := 60.0
const CRASH_LOG := "MeetingTranscriptions/engine-crash.log"
const PORT_FILE := "MeetingTranscriptions/engine-port"

var port: int = DEFAULT_PORT
var is_connected_to_engine := false

var _tcp := StreamPeerTCP.new()
var _buffer := PackedByteArray()
var _next_id := 1
var _callbacks := {}
var _retry_in := 0.0
var _failed_tries := 0
var _spawned_pid := -1
var _was_connected := false
var _spawns := 0
var _spawned_at := 0.0
var _spawn_unix := 0.0
var _last_progress := -1
var _gave_up := false
var _timeout_reported := false


func _ready() -> void:
	var env_port := OS.get_environment("MT_ENGINE_PORT")
	if env_port.is_valid_int():
		port = int(env_port)
	_try_connect()


func _process(delta: float) -> void:
	_tcp.poll()
	var status := _tcp.get_status()
	match status:
		StreamPeerTCP.STATUS_CONNECTED:
			if not is_connected_to_engine:
				is_connected_to_engine = true
				_tcp.set_no_delay(true)
				_was_connected = true
				_failed_tries = 0
				_spawned_at = 0.0
				connected.emit()
			_read_available()
		StreamPeerTCP.STATUS_CONNECTING:
			pass
		_:
			if is_connected_to_engine:
				is_connected_to_engine = false
				_buffer.clear()
				disconnected.emit()
			_watch_spawned_engine()
			_retry_in -= delta
			if _retry_in <= 0.0:
				_failed_tries += 1
				if _failed_tries >= SPAWN_AFTER_FAILED_TRIES and _spawned_pid < 0 and not _gave_up:
					_spawn_engine()
				_try_connect()


## While we wait for an engine we started: show progress, and notice when it
## exits or takes too long, instead of sitting on "Starting engine" forever.
func _watch_spawned_engine() -> void:
	if _spawned_pid < 0 or _spawned_at <= 0.0:
		return
	var waited := Time.get_ticks_msec() / 1000.0 - _spawned_at
	var port_file := OS.get_data_dir().path_join(PORT_FILE)
	if FileAccess.file_exists(port_file):
		# The engine may have had to pick another port than ours.
		var written := FileAccess.get_file_as_string(port_file).strip_edges()
		if written.is_valid_int() and int(written) != port:
			port = int(written)
			_try_connect()
	if not OS.is_process_running(_spawned_pid):
		var why := "The engine stopped right after starting."
		var crash := _read_crash_log()
		if crash != "":
			why += "\n" + crash
		why += "\nDetails: " + crash_log_path()
		_spawned_pid = -1
		_spawned_at = 0.0
		if _spawns >= MAX_SPAWNS:
			_gave_up = true
		engine_failed.emit(why)
		return
	var seconds := int(waited)
	if seconds != _last_progress and seconds > 0 and seconds % 3 == 0:
		_last_progress = seconds
		if waited < START_TIMEOUT_SECONDS:
			engine_log.emit("Starting engine... %ds (the first start can take a while)" % seconds)
		elif not _timeout_reported:
			_timeout_reported = true
			engine_failed.emit(("The engine is running but not answering on port %d after %ds. " +
				"A firewall or antivirus may be blocking it, or another app uses the port. " +
				"Details: %s") % [port, seconds, crash_log_path()])


static func crash_log_path() -> String:
	return OS.get_data_dir().path_join(CRASH_LOG)


## The last error the engine wrote, if it wrote one since we started it.
func _read_crash_log() -> String:
	var path := crash_log_path()
	if not FileAccess.file_exists(path) or FileAccess.get_modified_time(path) + 2 < int(_spawn_unix):
		return ""
	var text := FileAccess.get_file_as_string(path).strip_edges()
	var lines := text.split("\n")
	var tail: Array = []
	for i in range(lines.size() - 1, -1, -1):
		if lines[i].begins_with("--- "):
			break
		tail.push_front(lines[i])
	# The last line of a traceback is the actual error.
	return tail[-1] if not tail.is_empty() else ""


func _try_connect() -> void:
	_retry_in = CONNECT_RETRY_SECONDS
	_tcp = StreamPeerTCP.new()
	_tcp.connect_to_host("127.0.0.1", port)


func _read_available() -> void:
	var available := _tcp.get_available_bytes()
	if available <= 0:
		return
	var result: Array = _tcp.get_partial_data(available)
	if result[0] != OK:
		return
	_buffer.append_array(result[1])
	while true:
		var newline := _buffer.find(10)
		if newline < 0:
			break
		var line := _buffer.slice(0, newline).get_string_from_utf8()
		_buffer = _buffer.slice(newline + 1)
		if line.strip_edges().is_empty():
			continue
		var msg = JSON.parse_string(line)
		if typeof(msg) != TYPE_DICTIONARY:
			continue
		if msg.get("type") == "response":
			var id = msg.get("id")
			if id != null and _callbacks.has(int(id)):
				var cb: Callable = _callbacks[int(id)]
				_callbacks.erase(int(id))
				if cb.is_valid():
					cb.call(msg)
		event_received.emit(msg)


## Sends a command; `callback` gets the response dictionary ({ok, data, error}).
func send(cmd: String, args: Dictionary = {}, callback: Callable = Callable()) -> int:
	if not is_connected_to_engine:
		return -1
	var id := _next_id
	_next_id += 1
	var msg := args.duplicate()
	msg["cmd"] = cmd
	msg["id"] = id
	if callback.is_valid():
		_callbacks[id] = callback
	var data := (JSON.stringify(msg) + "\n").to_utf8_buffer()
	_tcp.put_data(data)
	return id


func _spawn_engine() -> void:
	var launch := _engine_command()
	if launch.is_empty():
		engine_log.emit("No engine found. Run the engine with: python -m engine")
		return
	var args: PackedStringArray = launch["args"]
	var port_file := OS.get_data_dir().path_join(PORT_FILE)
	DirAccess.remove_absolute(port_file)
	args.append_array(PackedStringArray(["--port", str(port), "--exit-when-alone", "--port-file", port_file]))
	if launch.has("pythonpath"):
		var existing := OS.get_environment("PYTHONPATH")
		var sep := ";" if OS.get_name() == "Windows" else ":"
		OS.set_environment("PYTHONPATH", launch["pythonpath"] + ((sep + existing) if existing != "" else ""))
		var env_file: String = launch["pythonpath"].path_join(".env")
		if FileAccess.file_exists(env_file):
			args.append_array(PackedStringArray(["--import-env", env_file]))
	# Same folder the UI reads it from (crash_log_path) on every platform.
	OS.set_environment("MT_ENGINE_CRASH_DIR", crash_log_path().get_base_dir())
	_spawns += 1
	_spawn_unix = Time.get_unix_time_from_system()
	_spawned_pid = OS.create_process(launch["cmd"], args, false)
	if _spawned_pid < 0:
		_gave_up = true
		engine_failed.emit("Could not start the engine: " + launch["cmd"])
		return
	_spawned_at = Time.get_ticks_msec() / 1000.0
	_last_progress = -1
	engine_log.emit("Starting engine...")
	print("Starting engine: %s %s (pid %d)" % [launch["cmd"], " ".join(args), _spawned_pid])


func _engine_command() -> Dictionary:
	var windows := OS.get_name() == "Windows"
	var override := OS.get_environment("MT_ENGINE_CMD")
	if override != "":
		return {"cmd": override, "args": PackedStringArray()}

	# Packaged build: engine sidecar next to the executable (also inside a macOS .app).
	var exe_dir := OS.get_executable_path().get_base_dir()
	var sidecar_name := "meeting-engine.exe" if windows else "meeting-engine"
	for candidate in [
		exe_dir.path_join("engine").path_join(sidecar_name),
		exe_dir.path_join(sidecar_name),
		exe_dir.path_join("../Resources/engine").path_join(sidecar_name),
	]:
		if FileAccess.file_exists(candidate):
			return {"cmd": candidate, "args": PackedStringArray()}

	# Running from source: the repository root is the parent of the Godot project.
	var repo_root := ProjectSettings.globalize_path("res://").trim_suffix("/").get_base_dir()
	if not DirAccess.dir_exists_absolute(repo_root.path_join("engine")):
		return {}
	var venv_python := repo_root.path_join(".venv/Scripts/python.exe" if windows else ".venv/bin/python")
	var python := venv_python if FileAccess.file_exists(venv_python) else ("python" if windows else "python3")
	return {"cmd": python, "args": PackedStringArray(["-m", "engine"]), "pythonpath": repo_root}
