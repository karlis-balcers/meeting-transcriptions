class_name EngineClient
extends Node
## Talks to the Python engine over a localhost socket (newline-delimited JSON).
## If no engine is listening it starts one: the packaged sidecar next to the
## app, or `python -m engine` from the repository when running from source.

signal connected
signal disconnected
signal event_received(msg: Dictionary)
signal engine_log(text: String)

const DEFAULT_PORT := 47321
const CONNECT_RETRY_SECONDS := 1.0
const SPAWN_AFTER_FAILED_TRIES := 2

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
				connected.emit()
			_read_available()
		StreamPeerTCP.STATUS_CONNECTING:
			pass
		_:
			if is_connected_to_engine:
				is_connected_to_engine = false
				_buffer.clear()
				disconnected.emit()
			_retry_in -= delta
			if _retry_in <= 0.0:
				_failed_tries += 1
				if _failed_tries == SPAWN_AFTER_FAILED_TRIES and _spawned_pid < 0:
					_spawn_engine()
				_try_connect()


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
	args.append_array(PackedStringArray(["--port", str(port), "--exit-when-alone"]))
	if launch.has("pythonpath"):
		var existing := OS.get_environment("PYTHONPATH")
		var sep := ";" if OS.get_name() == "Windows" else ":"
		OS.set_environment("PYTHONPATH", launch["pythonpath"] + ((sep + existing) if existing != "" else ""))
		var env_file: String = launch["pythonpath"].path_join(".env")
		if FileAccess.file_exists(env_file):
			args.append_array(PackedStringArray(["--import-env", env_file]))
	_spawned_pid = OS.create_process(launch["cmd"], args, false)
	engine_log.emit("Starting engine: %s %s (pid %d)" % [launch["cmd"], " ".join(args), _spawned_pid])


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
