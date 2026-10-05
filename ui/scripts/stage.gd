class_name Stage
extends Control
## The main view: every speaker on a ring around a live transcript.
## Node size follows talk share, the outer ring color is the speaker's mood,
## a glow pulses while they talk, lines show who answers whom, and check
## results float up from the speaker who triggered them.

signal speaker_clicked(name: String)

const MAX_TRANSCRIPT_PARAGRAPHS := 400
const BADGE_SECONDS := 7.0
const PARTICLE_SECONDS := 0.9

var speakers := {}          # name -> Dictionary
var order: Array = []
var transitions: Array = []
var badges: Array = []
var particles: Array = []
var me_name := "Me"
var selected := ""
var hovered := ""
var recording := false

var _transcript: RichTextLabel
var _header: Label
var _live: Label
var _empty_hint: Label
var _overlay: Control
var _center_rect := Rect2()
var _positions := {}
var _time := 0.0
var _lines := 0


func _ready() -> void:
	clip_contents = true
	mouse_filter = Control.MOUSE_FILTER_STOP
	_header = Label.new()
	_header.add_theme_color_override("font_color", Palette.TEXT_DIM)
	_header.add_theme_font_size_override("font_size", 13)
	_header.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_header.text = "LIVE TRANSCRIPT"
	add_child(_header)

	_transcript = RichTextLabel.new()
	_transcript.bbcode_enabled = true
	_transcript.scroll_following = true
	_transcript.selection_enabled = true
	_transcript.add_theme_font_size_override("normal_font_size", 16)
	_transcript.add_theme_font_size_override("bold_font_size", 16)
	_transcript.add_theme_color_override("default_color", Palette.TEXT)
	_transcript.add_theme_constant_override("line_separation", 4)
	add_child(_transcript)

	_live = Label.new()
	_live.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_live.add_theme_font_size_override("font_size", 13)
	_live.add_theme_color_override("font_color", Palette.TEXT_DIM)
	add_child(_live)

	_empty_hint = Label.new()
	_empty_hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_empty_hint.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_empty_hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_empty_hint.add_theme_color_override("font_color", Palette.TEXT_DIM)
	_empty_hint.text = "Press Start and talk. Everyone who speaks joins the circle."
	_empty_hint.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_empty_hint)

	_overlay = Control.new()
	_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_overlay.draw.connect(_draw_badges)
	add_child(_overlay)

	resized.connect(_layout)
	_layout()


func _process(delta: float) -> void:
	_time += delta
	var now := Time.get_ticks_msec() / 1000.0
	var live_names: Array[String] = []
	for name in speakers:
		var s: Dictionary = speakers[name]
		s.level = max(0.0, s.level - delta * 1.6)
		if s.level > 0.15:
			live_names.append(name)
	badges = badges.filter(func(b): return now - b.born < BADGE_SECONDS)
	particles = particles.filter(func(p): return now - p.born < PARTICLE_SECONDS)
	if not recording:
		_live.text = "Not recording"
	elif live_names.is_empty():
		_live.text = "Listening..."
	else:
		_live.text = "%s speaking" % ", ".join(live_names)
	queue_redraw()
	_overlay.queue_redraw()


# ------------------------------------------------------------------ public API

func reset() -> void:
	for name in speakers.keys():
		if name != me_name:
			speakers.erase(name)
	order = order.filter(func(n): return n == me_name)
	transitions.clear()
	badges.clear()
	particles.clear()
	_transcript.clear()
	_lines = 0
	if speakers.has(me_name):
		speakers[me_name].stats = {}
		speakers[me_name].mood = null
	_update_hint()


func set_me(name: String) -> void:
	if name == me_name and speakers.has(name):
		return
	if speakers.has(me_name) and me_name != name:
		rename(me_name, name)
	me_name = name
	ensure_speaker(name, true)


func set_title(text: String) -> void:
	_header.text = text.to_upper() if text != "" else "LIVE TRANSCRIPT"


func ensure_speaker(name: String, is_me := false) -> Dictionary:
	if not speakers.has(name):
		speakers[name] = {
			"name": name,
			"is_me": is_me or name == me_name,
			"color": Palette.speaker_color(name, is_me or name == me_name),
			"stats": {},
			"mood": null,
			"mood_reason": "",
			"level": 0.0,
			"hits": 0,
			"facts_wrong": 0,
			"utterances_seen": 0,
		}
		if name == me_name:
			order.push_front(name)
		else:
			order.append(name)
		_update_hint()
	return speakers[name]


func add_transcript(msg: Dictionary) -> void:
	var name: String = msg.get("speaker", "?")
	var s := ensure_speaker(name, bool(msg.get("is_me", false)))
	s.level = max(s.level, 0.6)
	s.utterances_seen += 1
	particles.append({"speaker": name, "born": Time.get_ticks_msec() / 1000.0, "color": s.color})
	var hex: String = s.color.to_html(false)
	var line := "[color=#5d647a]%s[/color]  [color=#%s][b]%s[/b][/color]  %s" % [
		Palette.clock(float(msg.get("at", 0.0))), hex, _escape(name), _escape(str(msg.get("text", "")))]
	if _lines > 0:
		_transcript.append_text("\n")
	_transcript.append_text(line)
	_lines += 1
	if _lines > MAX_TRANSCRIPT_PARAGRAPHS:
		_transcript.remove_paragraph(0)
		_lines -= 1
	_update_hint()


func set_level(name: String, rms: float, threshold: float) -> void:
	var t: float = max(threshold, 1.0)
	var level := clampf((rms - t) / (t * 5.0), 0.0, 1.0)
	if level <= 0.05 and not speakers.has(name):
		return
	var s := ensure_speaker(name)
	s.level = max(s.level, level)


func apply_stats(snapshot: Dictionary) -> void:
	var by_name: Dictionary = snapshot.get("speakers", {})
	for name in by_name:
		var s := ensure_speaker(name, bool(by_name[name].get("is_me", false)))
		s.stats = by_name[name]
		if by_name[name].get("mood") != null:
			s.mood = by_name[name].mood
	transitions = snapshot.get("transitions", [])


func apply_speaker_stats(name: String, stats: Dictionary) -> void:
	var s := ensure_speaker(name, bool(stats.get("is_me", false)))
	s.stats = stats
	if stats.get("mood") != null:
		s.mood = stats.mood


func add_check(msg: Dictionary) -> void:
	var name: String = msg.get("speaker", "")
	if name == "":
		return
	var s := ensure_speaker(name)
	var result: Dictionary = msg.get("result", {})
	var text := ""
	var color := Palette.ACCENT
	match str(msg.get("kind", "")):
		"mood":
			var changed: bool = s.mood != result.get("mood")
			s.mood = result.get("mood")
			s.mood_reason = str(result.get("reason", ""))
			if not changed:
				return
			text = str(result.get("mood", ""))
			color = Palette.mood_color(result.get("mood"))
		"fact":
			var verdict := str(result.get("verdict", ""))
			if verdict == "incorrect":
				s.facts_wrong += 1
			text = "%s: %s" % [verdict.capitalize(), str(result.get("claim", ""))]
			color = Palette.FACT_COLORS.get(verdict, Palette.WARN)
		_:
			s.hits += 1
			text = "%s: %s" % [str(msg.get("name", "")), str(result.get("label", ""))]
			var c = msg.get("color")
			color = Color(str(c)) if c != null and str(c) != "" else Palette.ACCENT
	if text.length() > 48:
		text = text.substr(0, 46) + "..."
	badges.append({"speaker": name, "text": text, "color": color, "born": Time.get_ticks_msec() / 1000.0})


func rename(old: String, new_name: String) -> void:
	if not speakers.has(old) or old == new_name:
		return
	var s: Dictionary = speakers[old]
	speakers.erase(old)
	order.erase(old)
	if speakers.has(new_name):
		var dst: Dictionary = speakers[new_name]
		dst.hits += s.hits
		dst.facts_wrong += s.facts_wrong
	else:
		s.name = new_name
		s.color = Palette.speaker_color(new_name, s.is_me)
		speakers[new_name] = s
		order.append(new_name)
	if selected == old:
		selected = new_name


func speaker_info(name: String) -> Dictionary:
	return speakers.get(name, {})


# ------------------------------------------------------------------- layout

func _layout() -> void:
	var w := size.x
	var h := size.y
	var radii := _orbit_radii()
	var rx := radii.x
	var ry := radii.y
	var half := Vector2(minf(w * 0.3, rx - 90.0), minf(h * 0.3, ry - 125.0))
	half = Vector2(maxf(half.x, 140.0), maxf(half.y, 90.0))
	_center_rect = Rect2(_orbit_center() - half, half * 2.0)
	_header.position = _center_rect.position + Vector2(16, 10)
	_header.size = Vector2(_center_rect.size.x - 32, 20)
	_transcript.position = _center_rect.position + Vector2(16, 34)
	_transcript.size = _center_rect.size - Vector2(32, 64)
	_live.position = Vector2(_center_rect.position.x + 16, _center_rect.end.y - 26)
	_live.size = Vector2(_center_rect.size.x - 32, 20)
	_empty_hint.position = _center_rect.position + Vector2(24, 34)
	_empty_hint.size = _center_rect.size - Vector2(48, 64)


func _orbit_center() -> Vector2:
	return Vector2(size.x * 0.5, size.y * 0.5 - 16.0)


func _orbit_radii() -> Vector2:
	return Vector2(maxf(size.x * 0.5 - 100.0, 120.0), maxf(size.y * 0.5 - 100.0, 100.0))


func _update_hint() -> void:
	if _empty_hint:
		_empty_hint.visible = order.size() <= 1 and _lines == 0


func _compute_positions() -> void:
	_positions.clear()
	var c := _orbit_center()
	var radii := _orbit_radii()
	var rx := radii.x
	var ry := radii.y
	var n := order.size()
	var others: Array[String] = []
	for name in order:
		if name == me_name:
			_positions[name] = c + Vector2(0, ry)
		else:
			others.append(name)
	var slots := others.size() + (1 if _positions.has(me_name) else 0)
	for i in others.size():
		var angle := PI / 2.0 + TAU * float(i + 1) / float(maxi(slots, 1))
		if not _positions.has(me_name):
			angle = -PI / 2.0 + TAU * float(i) / float(maxi(n, 1))
		_positions[others[i]] = c + Vector2(cos(angle) * rx, sin(angle) * ry)


func _radius(s: Dictionary) -> float:
	var share := float(s.stats.get("talk_share", 0.0))
	return 30.0 + 26.0 * sqrt(clampf(share, 0.0, 1.0))


# --------------------------------------------------------------------- draw

func _draw() -> void:
	_compute_positions()
	var c := _orbit_center()
	var font := get_theme_default_font()
	var now := Time.get_ticks_msec() / 1000.0

	# Faint orbit rings.
	var rx := _orbit_radii().x
	var ry := _orbit_radii().y
	for k in 3:
		var f := 0.75 + 0.25 * k
		_draw_ellipse(c, rx * f, ry * f, Color(1, 1, 1, 0.025 + 0.01 * k), 1.0)

	# Who answers whom.
	for t in transitions:
		var a: String = t.get("from", "")
		var b: String = t.get("to", "")
		if not (_positions.has(a) and _positions.has(b)):
			continue
		var pa: Vector2 = _positions[a]
		var pb: Vector2 = _positions[b]
		var col: Color = speakers[a].color.lerp(speakers[b].color, 0.5)
		col.a = 0.22
		var width := 1.0 + minf(float(t.get("count", 1)), 12.0) * 0.5
		_draw_curve(pa, pb, c, col, width)

	# Utterances flying into the transcript.
	for p in particles:
		if not _positions.has(p.speaker):
			continue
		var k := (now - float(p.born)) / PARTICLE_SECONDS
		var from: Vector2 = _positions[p.speaker]
		var to := _center_rect.get_center()
		var pos := from.lerp(to, ease(k, 0.6))
		var col: Color = p.color
		col.a = 1.0 - k
		draw_circle(pos, 5.0 * (1.0 - k) + 2.0, col)

	# Transcript card.
	var card := Palette.panel_style(Color(0.094, 0.106, 0.145, 0.92), 18, 0)
	if recording:
		card.border_color = Palette.RECORDING.lerp(Palette.BORDER, 0.5 + 0.5 * sin(_time * 2.0))
		card.set_border_width_all(2)
	draw_style_box(card, _center_rect)

	# Speakers.
	for name in order:
		if not _positions.has(name):
			continue
		_draw_speaker(speakers[name], _positions[name], font, now)



## Check badges go on an overlay so they sit above the transcript text.
func _draw_badges() -> void:
	var font := get_theme_default_font()
	var now := Time.get_ticks_msec() / 1000.0
	var c := _orbit_center()
	var stacks := {}
	for i in range(badges.size() - 1, -1, -1):
		var b: Dictionary = badges[i]
		if not _positions.has(b.speaker):
			continue
		var idx: int = stacks.get(b.speaker, 0)
		stacks[b.speaker] = idx + 1
		var age := now - float(b.born)
		var alpha := clampf(1.0 - (age - BADGE_SECONDS + 1.5) / 1.5, 0.0, 1.0)
		alpha = minf(alpha, clampf(age * 4.0, 0.0, 1.0))
		var base: Vector2 = _positions[b.speaker]
		var r := _radius(speakers[b.speaker])
		var text_w: float = font.get_string_size(str(b.text), HORIZONTAL_ALIGNMENT_LEFT, -1, 13).x
		var box_size := Vector2(text_w + 18, 22)
		var y := base.y - r + 4.0 - idx * 26.0
		var box := Rect2(Vector2(base.x + r + 10, y), box_size)
		if base.x < c.x - 1.0:
			box.position.x = base.x - r - 10 - box_size.x
		if box.position.x < 4.0 or box.end.x > size.x - 4.0:
			box.position = Vector2(base.x - box_size.x / 2.0, base.y - r - 34.0 - idx * 26.0)
		box.position.x = clampf(box.position.x, 4.0, size.x - box.size.x - 4.0)
		box.position.y = clampf(box.position.y, 4.0, size.y - box.size.y - 4.0)
		var sb := Palette.panel_style(Color(Palette.BG.lerp(b.color, 0.25), 0.95 * alpha), 11, 0)
		sb.border_color = Color(b.color.r, b.color.g, b.color.b, 0.9 * alpha)
		_overlay.draw_style_box(sb, box)
		_overlay.draw_string(font, box.position + Vector2(9, 16), str(b.text), HORIZONTAL_ALIGNMENT_LEFT, -1, 13,
			Color(Palette.TEXT, alpha))


func _draw_speaker(s: Dictionary, pos: Vector2, font: Font, now: float) -> void:
	var r := _radius(s)
	var base: Color = s.color
	var level: float = s.level

	# Talking glow.
	if level > 0.02:
		for k in 3:
			var phase := fmod(_time * 1.4 + k / 3.0, 1.0)
			var gr := r + 6.0 + phase * (10.0 + 22.0 * level)
			draw_arc(pos, gr, 0, TAU, 64, Color(base, (1.0 - phase) * 0.55 * level), 2.0, true)

	# Mood ring.
	var mood_col := Palette.mood_color(s.mood)
	draw_arc(pos, r + 4.0, 0, TAU, 64, mood_col, 5.0 if s.mood != null else 2.0, true)

	# Body.
	draw_circle(pos, r, base.darkened(0.55))
	draw_circle(pos, r - 3.0, base.darkened(0.35))
	if s.name == selected or s.name == hovered:
		draw_arc(pos, r + 10.0, 0, TAU, 64, Color(Palette.TEXT, 0.6 if s.name == selected else 0.3), 1.5, true)

	# Talk share as an arc inside the ring.
	var share := float(s.stats.get("talk_share", 0.0))
	if share > 0.0:
		draw_arc(pos, r - 1.5, -PI / 2.0, -PI / 2.0 + TAU * share, 48, base.lightened(0.25), 3.0, true)

	var ini := Palette.initials(s.name)
	var fs := int(clampf(r * 0.62, 16.0, 30.0))
	draw_string(font, pos + Vector2(-r, fs * 0.36), ini, HORIZONTAL_ALIGNMENT_CENTER, r * 2.0, fs, Palette.TEXT)

	# Name and a short stats line under the node.
	var label_w := 180.0
	var name_text: String = s.name + ("  (you)" if s.is_me else "")
	draw_string(font, pos + Vector2(-label_w / 2.0, r + 24.0), name_text, HORIZONTAL_ALIGNMENT_CENTER,
		label_w, 15, Palette.TEXT)
	var st: Dictionary = s.stats
	if not st.is_empty():
		var line := "%s talk  %d wpm" % [Palette.percent(float(st.get("talk_share", 0.0))), int(st.get("wpm", 0))]
		if int(st.get("questions", 0)) > 0:
			line += "  %d?" % int(st.questions)
		draw_string(font, pos + Vector2(-label_w / 2.0, r + 42.0), line, HORIZONTAL_ALIGNMENT_CENTER,
			label_w, 12, Palette.TEXT_DIM)
	if s.mood != null:
		draw_string(font, pos + Vector2(-label_w / 2.0, r + 58.0), str(s.mood), HORIZONTAL_ALIGNMENT_CENTER,
			label_w, 12, mood_col)

	# Small counters for check hits and wrong facts.
	var badge_pos := pos + Vector2(r * 0.72, -r * 0.72)
	if s.hits > 0:
		draw_circle(badge_pos, 10.0, Palette.ACCENT)
		draw_string(font, badge_pos + Vector2(-10, 4.5), str(s.hits), HORIZONTAL_ALIGNMENT_CENTER, 20, 12, Palette.BG)
	if s.facts_wrong > 0:
		var fp := pos + Vector2(-r * 0.72, -r * 0.72)
		draw_circle(fp, 10.0, Palette.BAD)
		draw_string(font, fp + Vector2(-10, 4.5), "!", HORIZONTAL_ALIGNMENT_CENTER, 20, 12, Palette.TEXT)


func _draw_ellipse(center: Vector2, rx: float, ry: float, color: Color, width: float) -> void:
	var pts := PackedVector2Array()
	for i in 97:
		var a := TAU * i / 96.0
		pts.append(center + Vector2(cos(a) * rx, sin(a) * ry))
	draw_polyline(pts, color, width, true)


func _draw_curve(a: Vector2, b: Vector2, pull: Vector2, color: Color, width: float) -> void:
	var ctrl := (a + b) * 0.5
	ctrl = ctrl.lerp(pull, 0.45)
	var pts := PackedVector2Array()
	for i in 25:
		var t := i / 24.0
		pts.append(a.lerp(ctrl, t).lerp(ctrl.lerp(b, t), t))
	draw_polyline(pts, color, width, true)


# -------------------------------------------------------------------- input

func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		var h := _speaker_at(event.position)
		if h != hovered:
			hovered = h
			mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND if h != "" else Control.CURSOR_ARROW
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		var name := _speaker_at(event.position)
		if name != "":
			selected = name
			speaker_clicked.emit(name)
			accept_event()


func _speaker_at(point: Vector2) -> String:
	for name in _positions:
		if point.distance_to(_positions[name]) <= _radius(speakers[name]) + 6.0:
			return name
	return ""


func _escape(text: String) -> String:
	return text.replace("[", "[lb]")
