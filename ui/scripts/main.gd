extends Control
## App shell: top bar, the speaker circle, sidebar and status bar. All real
## work happens in the Python engine; this script just routes its events.

var engine: EngineClient
var stage: Stage
var speaker_panel: SpeakerPanel
var settings_dialog: SettingsDialog
var profiles_dialog: ProfilesDialog

var settings := {}
var profiles := {}
var stats := {}
var recording := false
var muted := false
var _auto_started := false
var _llm_info := {}

var _start_btn: Button
var _mute_btn: Button
var _split_btn: Button
var _lang: OptionButton
var _mic: OptionButton
var _out: OptionButton
var _ai_btn: Button
var _status: Label
var _engine_error_dialog: AcceptDialog
var _path: Label
var _rec_dot: Label
var _feed: VBoxContainer
var _sidebar_tabs: TabContainer
var _devices := {}


func _ready() -> void:
	var bg := ColorRect.new()
	bg.color = Palette.BG
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(bg)

	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 10)
	add_child(margin)
	var root := VBoxContainer.new()
	root.add_theme_constant_override("separation", 10)
	margin.add_child(root)

	root.add_child(_build_top_bar())

	var body := HBoxContainer.new()
	body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	body.add_theme_constant_override("separation", 10)
	root.add_child(body)

	stage = Stage.new()
	stage.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	stage.size_flags_vertical = Control.SIZE_EXPAND_FILL
	stage.speaker_clicked.connect(_on_speaker_clicked)
	body.add_child(stage)
	body.add_child(_build_sidebar())

	root.add_child(_build_status_bar())

	settings_dialog = SettingsDialog.new()
	settings_dialog.save_requested.connect(func(values): engine.send("save_settings", {"settings": values}))
	settings_dialog.llm_action.connect(_on_llm_action)
	add_child(settings_dialog)

	profiles_dialog = ProfilesDialog.new()
	profiles_dialog.delete_requested.connect(func(n): engine.send("delete_profile", {"name": n}))
	profiles_dialog.rename_requested.connect(func(a, b): engine.send("rename_profile", {"old": a, "new": b}))
	add_child(profiles_dialog)

	engine = EngineClient.new()
	engine.connected.connect(_on_connected)
	engine.disconnected.connect(_on_disconnected)
	engine.event_received.connect(_on_event)
	engine.engine_log.connect(func(t): _set_status(t, "info"))
	engine.engine_failed.connect(_on_engine_failed)
	add_child(engine)
	_set_status("Connecting to the engine...", "info")
	_update_controls()


# ----------------------------------------------------------------------- UI

func _build_top_bar() -> Control:
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", Palette.panel_style(Palette.PANEL, 12, 8))
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	panel.add_child(row)

	_rec_dot = Label.new()
	_rec_dot.text = "●"
	_rec_dot.add_theme_font_size_override("font_size", 18)
	row.add_child(_rec_dot)
	var title := Label.new()
	title.text = "Meeting Transcriptions"
	title.add_theme_font_size_override("font_size", 17)
	row.add_child(title)
	row.add_child(VSeparator.new())

	_start_btn = Button.new()
	_start_btn.custom_minimum_size = Vector2(96, 0)
	_start_btn.pressed.connect(_toggle_recording)
	row.add_child(_start_btn)

	_mute_btn = Button.new()
	_mute_btn.toggle_mode = true
	_mute_btn.tooltip_text = "Mute your microphone"
	_mute_btn.toggled.connect(func(on): engine.send("mute", {"muted": on}))
	row.add_child(_mute_btn)

	_split_btn = Button.new()
	_split_btn.text = "Split"
	_split_btn.tooltip_text = "Cut the current audio chunk now (key S)"
	_split_btn.pressed.connect(func(): engine.send("split"))
	row.add_child(_split_btn)

	_lang = OptionButton.new()
	_lang.tooltip_text = "Transcription language"
	row.add_child(_lang)

	_mic = OptionButton.new()
	_mic.tooltip_text = "Microphone"
	_mic.custom_minimum_size = Vector2(170, 0)
	_mic.clip_text = true
	_mic.item_selected.connect(func(i): _select_device("input", _mic, i))
	row.add_child(_mic)
	_out = OptionButton.new()
	_out.tooltip_text = "Output capture (what the other side says)"
	_out.custom_minimum_size = Vector2(170, 0)
	_out.clip_text = true
	_out.item_selected.connect(func(i): _select_device("output", _out, i))
	row.add_child(_out)
	var refresh := Button.new()
	refresh.text = "Refresh"
	refresh.tooltip_text = "Reload audio devices"
	refresh.pressed.connect(_load_devices)
	row.add_child(refresh)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(spacer)

	_ai_btn = Button.new()
	_ai_btn.tooltip_text = "Local AI for mood, fact checks and your own checks"
	_ai_btn.pressed.connect(func(): _open_settings("Local AI"))
	row.add_child(_ai_btn)
	var profiles_btn := Button.new()
	profiles_btn.text = "Profiles"
	profiles_btn.pressed.connect(func():
		engine.send("profiles")
		profiles_dialog.open())
	row.add_child(profiles_btn)
	var settings_btn := Button.new()
	settings_btn.text = "Settings"
	settings_btn.pressed.connect(func(): _open_settings(""))
	row.add_child(settings_btn)
	return panel


func _build_sidebar() -> Control:
	var panel := PanelContainer.new()
	panel.custom_minimum_size = Vector2(360, 0)
	panel.add_theme_stylebox_override("panel", Palette.panel_style(Palette.PANEL, 12, 10))
	_sidebar_tabs = TabContainer.new()
	panel.add_child(_sidebar_tabs)

	var speaker_scroll := ScrollContainer.new()
	speaker_scroll.name = "Speaker"
	speaker_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	speaker_panel = SpeakerPanel.new()
	speaker_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	speaker_panel.rename_requested.connect(func(a, b): engine.send("rename_speaker", {"old": a, "new": b}))
	speaker_scroll.add_child(speaker_panel)
	_sidebar_tabs.add_child(speaker_scroll)

	var feed_scroll := ScrollContainer.new()
	feed_scroll.name = "Insights"
	feed_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	_feed = VBoxContainer.new()
	_feed.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_feed.add_theme_constant_override("separation", 6)
	feed_scroll.add_child(_feed)
	_sidebar_tabs.add_child(feed_scroll)
	_add_feed_entry(null, "Fact checks and your custom checks show up here while people talk.", Palette.TEXT_DIM)
	_feed.get_child(0).set_meta("placeholder", true)
	return panel


func _build_status_bar() -> Control:
	var row := HBoxContainer.new()
	_status = Label.new()
	_status.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_status.clip_text = true
	row.add_child(_status)
	_path = Label.new()
	_path.add_theme_color_override("font_color", Palette.TEXT_DIM)
	_path.add_theme_font_size_override("font_size", 12)
	row.add_child(_path)
	var open_folder := Button.new()
	open_folder.text = "Open folder"
	open_folder.flat = true
	open_folder.pressed.connect(func():
		if settings.has("output_dir"):
			OS.shell_open(str(settings.output_dir)))
	row.add_child(open_folder)
	return row


func _update_controls() -> void:
	var online := engine != null and engine.is_connected_to_engine
	_start_btn.disabled = not online
	_start_btn.text = "Stop" if recording else "Start"
	_start_btn.modulate = Palette.RECORDING.lightened(0.3) if recording else Color.WHITE
	_mute_btn.disabled = not online
	_mute_btn.set_pressed_no_signal(muted)
	_mute_btn.text = "Mic muted" if muted else "Mic on"
	_split_btn.disabled = not recording
	_lang.disabled = recording
	_rec_dot.add_theme_color_override("font_color",
		Palette.RECORDING if recording else (Palette.GOOD if online else Palette.TEXT_DIM))
	stage.recording = recording
	_update_ai_button()


func _update_ai_button() -> void:
	var text := "Local AI: off"
	var color := Palette.TEXT_DIM
	if bool(settings.get("llm_enabled", false)):
		if _llm_info.get("model_ready", false):
			text = "Local AI: on"
			color = Palette.GOOD
		elif _llm_info.get("running", false):
			text = "Local AI: no model"
			color = Palette.WARN
		elif _llm_info.get("installed", false):
			text = "Local AI: not running"
			color = Palette.WARN
		else:
			text = "Local AI: install"
			color = Palette.WARN
	_ai_btn.text = text
	_ai_btn.add_theme_color_override("font_color", color)


func _set_status(text: String, level: String) -> void:
	_status.text = text
	var color := Palette.TEXT
	if level == "warning":
		color = Palette.WARN
	elif level == "error":
		color = Palette.BAD
	_status.add_theme_color_override("font_color", color)


func _add_feed_entry(speaker, text: String, color: Color) -> void:
	if _feed.get_child_count() == 1 and _feed.get_child(0).has_meta("placeholder"):
		_feed.get_child(0).free()
	var panel := PanelContainer.new()
	var sb := Palette.panel_style(Palette.PANEL_LIGHT, 8, 8)
	sb.border_color = color
	sb.border_width_left = 3
	panel.add_theme_stylebox_override("panel", sb)
	var label := RichTextLabel.new()
	label.bbcode_enabled = true
	label.fit_content = true
	label.selection_enabled = true
	label.scroll_active = false
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	if speaker != null:
		var sc := Palette.speaker_color(str(speaker), str(speaker) == stage.me_name)
		label.append_text("[color=#%s][b]%s[/b][/color]  " % [sc.to_html(false), str(speaker).replace("[", "[lb]")])
	label.append_text(text)
	panel.add_child(label)
	_feed.add_child(panel)
	_feed.move_child(panel, 0)
	while _feed.get_child_count() > 150:
		_feed.get_child(_feed.get_child_count() - 1).free()


# ------------------------------------------------------------------ actions

func _toggle_recording() -> void:
	if recording:
		engine.send("stop")
		_set_status("Stopping, finishing the last chunks...", "info")
	else:
		var lang := ""
		if _lang.item_count > 0:
			lang = _lang.get_item_text(_lang.selected)
		stage.reset()
		engine.send("start", {"language": lang})
		_set_status("Starting...", "info")


func _open_settings(tab: String) -> void:
	if settings.is_empty():
		return
	settings_dialog.open_with(settings)
	if tab != "":
		var tabs: TabContainer = settings_dialog._tabs
		for i in tabs.get_tab_count():
			if tabs.get_tab_title(i) == tab:
				tabs.current_tab = i
	engine.send("llm_status")


func _on_llm_action(action: String) -> void:
	settings_dialog.set_llm_status("Working...", -1.0)
	engine.send(action, {}, func(resp):
		var data = resp.get("data")
		if not resp.get("ok", false) or (data is Dictionary and data.get("ok") == false):
			var err = resp.get("error") if not resp.get("ok", false) else data.get("error")
			settings_dialog.set_llm_status("Error: " + str(err))
		elif action == "llm_start":
			settings_dialog.set_llm_status("Starting the local AI, press Check in a moment."))


func _select_device(kind: String, button: OptionButton, index: int) -> void:
	var dev_index = button.get_item_metadata(index)
	var list: Array = _devices.get("inputs" if kind == "input" else "outputs", [])
	var name := ""
	for d in list:
		if d.get("index") == dev_index:
			name = str(d.get("name", ""))
	engine.send("select_device", {"kind": kind, "index": dev_index, "name": name})
	if recording:
		_set_status("New device is used from the next Start.", "info")


func _load_devices() -> void:
	engine.send("devices", {}, func(resp):
		if not resp.get("ok", false):
			_set_status("Could not list audio devices: " + str(resp.get("error", "")), "error")
			return
		_devices = resp.get("data", {})
		_fill_devices(_mic, _devices.get("inputs", []), _devices.get("selected_input"))
		_fill_devices(_out, _devices.get("outputs", []), _devices.get("selected_output"))
		if _devices.has("error"):
			_set_status(str(_devices.error), "error"))


func _fill_devices(button: OptionButton, list: Array, selected) -> void:
	button.clear()
	for d in list:
		button.add_item(str(d.get("label", d.get("name", "?"))))
		button.set_item_metadata(button.item_count - 1, d.get("index"))
		button.set_item_tooltip(button.item_count - 1, str(d.get("name", "")))
		if d.get("index") == selected:
			button.select(button.item_count - 1)
	if list.is_empty():
		button.add_item("No devices")
		button.disabled = true
	else:
		button.disabled = false


func _on_speaker_clicked(name: String) -> void:
	_sidebar_tabs.current_tab = 0
	_refresh_speaker_panel(name)


func _refresh_speaker_panel(name: String = "") -> void:
	if name == "":
		name = speaker_panel.speaker_name
	if name == "":
		return
	var live := stage.speaker_info(name)
	var st: Dictionary = stats.get("speakers", {}).get(name, live.get("stats", {}))
	speaker_panel.show_speaker(name, live, st, profiles.get(name, {}))


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_S and recording:
		engine.send("split")
		_set_status("Split", "info")
		get_viewport().set_input_as_handled()


# ------------------------------------------------------------------- engine

func _on_engine_failed(text: String) -> void:
	_set_status(text.get_slice("\n", 0), "error")
	if _engine_error_dialog == null:
		_engine_error_dialog = AcceptDialog.new()
		_engine_error_dialog.title = "Engine problem"
		add_child(_engine_error_dialog)
	_engine_error_dialog.dialog_text = text
	_engine_error_dialog.popup_centered(Vector2i(640, 0))


func _on_connected() -> void:
	_set_status("Engine connected", "info")
	_update_controls()


func _on_disconnected() -> void:
	recording = false
	_set_status("Engine disconnected, reconnecting...", "warning")
	_update_controls()


func _apply_settings(values: Dictionary) -> void:
	settings = values
	stage.set_me(str(values.get("your_name", "Me")))
	var current := _lang.get_item_text(_lang.selected) if _lang.item_count > 0 else ""
	_lang.clear()
	for code in str(values.get("languages", "en")).split(",", false):
		_lang.add_item(code.strip_edges())
		if code.strip_edges() == current:
			_lang.select(_lang.item_count - 1)
	_lang.visible = _lang.item_count > 1
	_update_ai_button()


func _on_event(msg: Dictionary) -> void:
	match str(msg.get("type", "")):
		"hello":
			_apply_settings(msg.get("settings", {}))
			_apply_state(msg.get("state", {}))
			stats = msg.get("stats", {})
			stage.apply_stats(stats)
			_load_devices()
			engine.send("llm_status")
			# A setup task may still be running from before the window was reopened.
			engine.send("llm_log", {}, func(resp):
				var data: Dictionary = resp.get("data", {})
				settings_dialog.set_llm_log(data.get("lines", []))
				if data.get("busy", false):
					settings_dialog.set_llm_busy(str(data.get("message", "")), float(data.get("progress", -1.0)),
						float(data.get("elapsed", 0))))
			engine.send("profiles", {}, func(resp):
				if resp.get("ok", false):
					_set_profiles(resp.get("data", {})))
			if not settings.get("openai_api_key_set", false):
				_set_status("Add your OpenAI API key in Settings to start.", "warning")
			if bool(settings.get("auto_start", false)) and not recording and not _auto_started:
				_auto_started = true
				_toggle_recording()
		"state":
			_apply_state(msg)
		"status":
			_set_status(str(msg.get("message", "")), str(msg.get("level", "info")))
		"settings":
			_apply_settings(msg.get("settings", {}))
			engine.send("llm_status")
		"transcript":
			stage.add_transcript(msg)
		"level":
			stage.set_level(str(msg.get("speaker", "")), float(msg.get("rms", 0.0)), float(msg.get("threshold", 50.0)))
		"active_speaker":
			pass
		"stats":
			stats = msg.get("stats", {})
			stage.apply_stats(stats)
			_refresh_speaker_panel()
		"speaker_stats":
			var name := str(msg.get("speaker", ""))
			if stats.has("speakers"):
				stats.speakers[name] = msg.get("stats", {})
			stage.apply_speaker_stats(name, msg.get("stats", {}))
			_refresh_speaker_panel()
		"speaker_renamed":
			stage.rename(str(msg.old), str(msg.new))
			if speaker_panel.speaker_name == str(msg.old):
				_refresh_speaker_panel(str(msg.new))
		"check":
			_on_check(msg)
		"llm":
			_on_llm(msg)
		"profiles":
			_set_profiles(msg.get("profiles", {}))


func _apply_state(state: Dictionary) -> void:
	if state.is_empty():
		return
	recording = bool(state.get("recording", false))
	muted = bool(state.get("muted", false))
	var path = state.get("transcript_path")
	_path.text = str(path) if path != null else ""
	var title = state.get("meeting_title")
	stage.set_title(str(title) if title != null else "")
	_update_controls()


func _on_check(msg: Dictionary) -> void:
	stage.add_check(msg)
	var result: Dictionary = msg.get("result", {})
	match str(msg.get("kind", "")):
		"mood":
			_refresh_speaker_panel()
		"fact":
			var verdict := str(result.get("verdict", ""))
			var color: Color = Palette.FACT_COLORS.get(verdict, Palette.WARN)
			var text := "[color=#%s]Fact check: %s[/color]\n%s" % [color.to_html(false), verdict, str(result.get("claim", "")).replace("[", "[lb]")]
			if str(result.get("note", "")) != "":
				text += "\n[color=#8b91a7]%s[/color]" % str(result.note).replace("[", "[lb]")
			_add_feed_entry(msg.get("speaker"), text, color)
		_:
			var c = msg.get("color")
			var color := Color(str(c)) if c != null and str(c) != "" else Palette.ACCENT
			var text := "[color=#%s]%s: %s[/color]" % [color.to_html(false), str(msg.get("name", "")), str(result.get("label", "")).replace("[", "[lb]")]
			if str(result.get("note", "")) != "":
				text += "\n[color=#8b91a7]%s[/color]" % str(result.note).replace("[", "[lb]")
			_add_feed_entry(msg.get("speaker"), text, color)


func _on_llm(msg: Dictionary) -> void:
	match str(msg.get("state", "")):
		"status":
			_llm_info = msg
			var text := ""
			if msg.get("model_ready", false):
				text = "Ready: %s at %s" % [msg.get("model", ""), msg.get("base_url", "")]
			elif msg.get("running", false):
				text = "Server is running but model %s is not downloaded. Press Download model." % msg.get("model", "")
			elif msg.get("starting", false):
				text = "Starting %s..." % _llm_name(msg)
			elif msg.get("installed", false):
				text = "%s is installed but not running. Press Start." % _llm_name(msg)
			else:
				text = "%s is not installed. Press Install to set it up." % _llm_name(msg)
			settings_dialog.set_llm_info(text)
			_update_ai_button()
		"busy":
			settings_dialog.set_llm_busy(str(msg.get("message", "")), float(msg.get("progress", -1.0)),
				float(msg.get("elapsed", 0)))
			_ai_btn.text = "Local AI: setting up..."
		"log":
			settings_dialog.append_llm_log(str(msg.get("line", "")))
		"done":
			settings_dialog.set_llm_status(str(msg.get("message", "")))
			_set_status(str(msg.get("message", "")), "info")
		"error":
			settings_dialog.set_llm_status("Error: " + str(msg.get("message", "")) + "  (details in the setup log below)")
			settings_dialog.show_llm_log()
			_set_status("Local AI: " + str(msg.get("message", "")), "error")


func _llm_name(info: Dictionary) -> String:
	match str(info.get("api", "")):
		"laya":
			return "Laya"
		"ollama":
			return "Ollama"
	return "The local AI server"


func _set_profiles(p: Dictionary) -> void:
	profiles = p
	profiles_dialog.set_profiles(p)
	_refresh_speaker_panel()
