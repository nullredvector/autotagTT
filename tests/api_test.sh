#!/bin/bash
# Integration test of /api/autotag* against a scratch archive server. usage: api_test.sh <port> <archive_dir>
P=$1; A=$2; B=http://localhost:$P; fail=0
ID1=7000000000000000009; ID2=7000000000000000010; ID3=7000000000000000011
chk() { if [ "$2" == "$3" ]; then echo "PASS $1"; else echo "FAIL $1: got $2 want $3"; fail=1; fi; }
post() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/feedback" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
STARS_BEFORE=$(shasum $A/data/.appdata/stars.json)
chk "GET lists 14 pending" "$(curl -s "$B/api/autotag?tag=braces" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["pending"]))')" 14
chk "GET bad tag -> 400" "$(curl -s -o /dev/null -w '%{http_code}' "$B/api/autotag?tag=%3Cb%3E")" 400
chk "GET unknown tag -> empty" "$(curl -s "$B/api/autotag?tag=zzz" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["pending"]))')" 0
chk "reject ok" "$(post '{"tag":"braces","id":"'$ID1'","decision":"rejected"}')" 200
chk "reject repeat idempotent" "$(post '{"tag":"braces","id":"'$ID1'","decision":"rejected"}')" 200
chk "flip without change -> 409" "$(post '{"tag":"braces","id":"'$ID1'","decision":"accepted"}')" 409
chk "restore with change" "$(post '{"tag":"braces","id":"'$ID1'","decision":"pending","change":true}')" 200
chk "accept ok" "$(post '{"tag":"braces","id":"'$ID2'","decision":"accepted"}')" 200
chk "undo accepted refused" "$(post '{"tag":"braces","id":"'$ID2'","decision":"rejected","change":true}')" 409
chk "unknown id -> 404" "$(post '{"tag":"braces","id":"123","decision":"accepted"}')" 404
chk "malformed id -> 400" "$(post '{"tag":"braces","id":"../../x","decision":"accepted"}')" 400
chk "bad decision -> 400" "$(post '{"tag":"braces","id":"'$ID3'","decision":"delete"}')" 400
chk "bad tag -> 400" "$(post '{"tag":"<x>","id":"'$ID3'","decision":"rejected"}')" 400
chk "invalid JSON -> 400" "$(post '{nope')" 400
chk "array body -> 400" "$(post '[1]')" 400
chk "non-JSON content type -> 415" "$(curl -s -o /dev/null -w '%{http_code}' -X POST $B/api/autotag/feedback -H 'Content-Type: text/plain' -d '{}')" 415
chk "cross-origin -> 403" "$(post '{"tag":"braces","id":"'$ID3'","decision":"rejected"}' -H 'Origin: http://evil.example')" 403
chk "same-origin ok" "$(post '{"tag":"braces","id":"'$ID3'","decision":"rejected"}' -H "Origin: $B")" 200
chk "oversize body -> 413" "$(post "$(python3 -c 'print("{\"x\":\""+"a"*5000+"\"}")')")" 413
chk "GET on feedback -> 405" "$(curl -s -o /dev/null -w '%{http_code}' $B/api/autotag/feedback)" 405
chk "PUT on feedback -> 405" "$(curl -s -o /dev/null -w '%{http_code}' -X PUT $B/api/autotag/feedback)" 405
chk "unsure ok" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"unsure"}')" 200
chk "unsure->accepted needs change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"accepted"}')" 409
chk "unsure->rejected with change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"rejected","change":true}')" 200
chk "rejected->unsure with change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"unsure","change":true}')" 200
chk "unsure->pending with change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"pending","change":true}')" 200
chk "unsure again (left set aside)" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"unsure"}')" 200
chk "pending now 12" "$(curl -s "$B/api/autotag?tag=braces" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["pending"]),len(d["accepted"]),len(d["rejected"]));assert len(d["unsure"])==1')" "11 1 1"
chk "feedback file mode 600" "$(stat -f %Lp $A/data/.appdata/autotag_feedback.json 2>/dev/null || stat -c %a $A/data/.appdata/autotag_feedback.json)" 600
chk "server never touched stars.json" "$(shasum $A/data/.appdata/stars.json)" "$STARS_BEFORE"
# ── AI tags + tag picker ──
ai() { curl -s "$B/api/autotag/ai" | python3 -c "$1"; }
chk "AI endpoint: has videos" "$(ai 'import sys,json;d=json.load(sys.stdin);print(d["available"],len(d["videos"])>0)')" "True True"
chk "AI hides video already in same-name human group" "$(ai 'import sys,json;d=json.load(sys.stdin);print("7000000000000000005" in d["videos"])')" False
chk "AI hides decided (rejected id3 braces)" "$(ai 'import sys,json;d=json.load(sys.stdin);print([t for t,_ in d["videos"].get("'$ID3'",[])])')" "['scenery']"
chk "AI-only tag feedback accepted (scenery on AI video)" "$(post '{"tag":"scenery","id":"7000000000000000012","decision":"rejected"}')" 200
chk "AI feedback on a tag it was not tagged with -> 404" "$(post '{"tag":"scenery","id":"7000000000000000040","decision":"rejected"}')" 404
chk "tags GET defaults to braces" "$(curl -s $B/api/autotag/tags)" '{"enabled":["braces"]}'
tp() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/tags" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
chk "tags set (case-insensitive, deduped)" "$(tp '{"enabled":["Braces","SCENERY","scenery"]}')" 200
chk "tags saved" "$(curl -s $B/api/autotag/tags)" '{"enabled":["braces","scenery"]}'
chk "tags unknown group -> 404" "$(tp '{"enabled":["no such group"]}')" 404
chk "tags non-array -> 400" "$(tp '{"enabled":"braces"}')" 400
chk "tags bad entry -> 400" "$(tp '{"enabled":[5]}')" 400
chk "tags cross-origin -> 403" "$(tp '{"enabled":["braces"]}' -H 'Origin: http://evil.example')" 403
chk "tags config mode 600" "$(stat -f %Lp $A/data/.appdata/autotag_tags.json 2>/dev/null || stat -c %a $A/data/.appdata/autotag_tags.json)" 600
chk "stars.json still untouched" "$(shasum $A/data/.appdata/stars.json)" "$STARS_BEFORE"
# ── run control (a fake worker is simulated by writing status.json) ──
AT=$A/data/.appdata/autotag
rp() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/run" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
setst() { python3 -c "import json,time,sys;d=json.loads(sys.argv[1]);d['heartbeat']=time.time()-float(sys.argv[2]);json.dump(d,open('$AT/status.json','w'))" "$1" "${2:-0}"; }
rm -f $AT/status.json $AT/run_request.json
chk "run: no worker -> 503" "$(rp '{}')" 503
setst '{"state":"idle"}' 120
chk "run: stale heartbeat -> 503" "$(rp '{}')" 503
setst '{"state":"idle"}' 0
chk "run: status shows idle+online" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["state"],d["workerOnline"])')" "idle True"
chk "run: cross-origin -> 403" "$(rp '{}' -H 'Origin: http://evil.example')" 403
chk "run: ok" "$(rp '{}')" 200
chk "run: request file mode 600" "$(stat -f %Lp $AT/run_request.json 2>/dev/null || stat -c %a $AT/run_request.json)" 600
chk "run: now queued" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["state"])')" queued
chk "run: second request while queued -> 409" "$(rp '{}')" 409
RID=$(python3 -c "import json;print(json.load(open('$AT/run_request.json'))['id'])")
setst '{"state":"running","phase":"embedding","last_request_id":"'$RID'"}' 0
chk "run: running" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["state"],d["phase"])')" "running embedding"
chk "run: while running -> 409" "$(rp '{}')" 409
setst '{"state":"idle","phase":"done","last_request_id":"'$RID'"}' 0
chk "run: too soon after previous -> 429" "$(rp '{}')" 429
chk "run: array body -> 400" "$(rp '[1]')" 400
chk "run: GET status has no paths/secrets" "$(curl -s $B/api/autotag/run | grep -c '/mnt\|/archive')" 0
sp() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/schedule" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
chk "schedule: ok" "$(sp '{"enabled":true,"hour":3}')" 200
chk "schedule: read back" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["schedule"])')" "{'enabled': True, 'hour': 3}"
chk "schedule: file mode 600" "$(stat -f %Lp $AT/schedule.json 2>/dev/null || stat -c %a $AT/schedule.json)" 600
chk "schedule: hour 24 -> 400" "$(sp '{"enabled":true,"hour":24}')" 400
chk "schedule: non-bool -> 400" "$(sp '{"enabled":"yes","hour":3}')" 400
chk "schedule: string hour -> 400" "$(sp '{"enabled":true,"hour":"3"}')" 400
chk "schedule: cross-origin -> 403" "$(sp '{"enabled":true,"hour":3}' -H 'Origin: http://evil.example')" 403
nt() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/notify-test" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
setst '{"state":"idle","notifications":false}' 0
chk "notify-test: notifications off -> 409" "$(nt '{}')" 409
setst '{"state":"idle","notifications":true}' 120
chk "notify-test: worker offline -> 503" "$(nt '{}')" 503
setst '{"state":"idle","notifications":true}' 0
chk "notify-test: cross-origin -> 403" "$(nt '{}' -H 'Origin: http://evil.example')" 403
chk "notify-test: ok" "$(nt '{}')" 200
chk "notify-test: file mode 600" "$(stat -f %Lp $AT/notify_test.json 2>/dev/null || stat -c %a $AT/notify_test.json)" 600
chk "notify-test: too soon -> 429" "$(nt '{}')" 429
chk "notify-test: status flag exposed" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["notifications"])')" True
rm -f $AT/notify_test.json
fp() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/frames" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
chk "frames: default config is off" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["frames"]["config"])')" "{'mode': 'off', 'mouth': False, 'text': True, 'budget_min': 120}"
chk "frames: set trial" "$(fp '{"mode":"trial","mouth":true,"budget_min":60}')" 200
chk "frames: read back" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["frames"]["config"])')" "{'mode': 'trial', 'mouth': True, 'text': True, 'budget_min': 60}"
chk "frames: file mode 600" "$(stat -f %Lp $AT/frames.json 2>/dev/null || stat -c %a $AT/frames.json)" 600
chk "frames: budget defaults" "$(fp '{"mode":"full","mouth":false}')" 200
chk "frames: bad mode -> 400" "$(fp '{"mode":"everything","mouth":false}')" 400
chk "frames: mouth must be boolean -> 400" "$(fp '{"mode":"full","mouth":"yes"}')" 400
chk "frames: budget too small -> 400" "$(fp '{"mode":"full","mouth":false,"budget_min":5}')" 400
chk "frames: budget string -> 400" "$(fp '{"mode":"full","mouth":false,"budget_min":"60"}')" 400
chk "frames: cross-origin -> 403" "$(fp '{"mode":"full","mouth":false}' -H 'Origin: http://evil.example')" 403
setst '{"state":"idle","frames":{"stats":{"done":120,"total":1000,"rate":2.5,"mode":"trial","ts":1},"trial":{"scenery":{"n_pos":40,"n":300,"ap_cover":0.2,"ap_frames":0.5}}}}' 0
chk "frames: stats and trial exposed" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;f=json.load(sys.stdin)["frames"];print(f["stats"]["done"],f["trial"]["scenery"]["ap_frames"])')" "120 0.5"
wp() { python3 -c "import json,time,sys;d=json.loads(sys.argv[1]);d.setdefault('ts',time.time()-float(sys.argv[2]));json.dump(d,open('$AT/frames_progress.json','w'))" "$1" "${2:-0}"; }
fprog() { curl -s $B/api/autotag/run | python3 -c "$1"; }
wp '{"mode":"full","total":58000,"covered":29000,"session_done":500,"session_target":2000,"rate":12.5,"running":true}' 0
chk "progress: live values exposed" "$(fprog 'import sys,json;p=json.load(sys.stdin)["frames"]["progress"];print(p["mode"],p["total"],p["covered"],p["sessionDone"],p["running"],p["threshold"])')" "full 58000 29000 500 True 0.95"
wp '{"mode":"full","total":58000,"covered":29000,"running":true}' 600
chk "progress: stale report is not 'running'" "$(fprog 'import sys,json;print(json.load(sys.stdin)["frames"]["progress"]["running"])')" False
wp '{"mode":"bogus","total":"x","covered":-5,"session_done":"a","rate":-1,"running":"yes"}' 0
chk "progress: junk is sanitised" "$(fprog 'import sys,json;p=json.load(sys.stdin)["frames"]["progress"];print(p["mode"],p["total"],p["covered"],p["sessionDone"],p["rate"],p["running"])')" "off 0 0 0 0 False"
wp '{"mode":"full","total":100,"covered":500,"running":false}' 0
chk "progress: covered never exceeds total" "$(fprog 'import sys,json;p=json.load(sys.stdin)["frames"]["progress"];print(p["covered"])')" 100
rm -f $AT/frames_progress.json
chk "progress: absent file -> null" "$(fprog 'import sys,json;print(json.load(sys.stdin)["frames"]["progress"])')" None
chk "progress: no paths in the status" "$(curl -s $B/api/autotag/run | grep -c '/mnt\\|/archive')" 0
rm -f $AT/frames.json
chk "frames: captions can be switched off" "$(fp '{"mode":"off","mouth":false,"text":false}')" 200
chk "frames: captions flag read back" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["frames"]["config"]["text"])')" False
chk "frames: text must be boolean -> 400" "$(fp '{"mode":"off","mouth":false,"text":"no"}')" 400
rm -f $AT/frames.json
# weak tags: review-only lists are listed and flagged, hopeless ones are hidden
cp $AT/autotag.json /tmp/at_keep.json
python3 - <<PY
import json
d={"_meta":{"metrics":{"braces":{"gate":"ok"},"scenery":{"gate":"weak","review_only":True,"verdict":"review_only"},"hopeless":{"gate":"weak","review_only":False,"verdict":"hopeless"}}},
   "braces":[{"id":"7000000000000000009","score":2.0}],"scenery":[{"id":"7000000000000000030","score":0.5},{"id":"7000000000000000031","score":0.4}],"hopeless":[]}
json.dump(d,open("$AT/autotag.json","w"))
PY
lc() { curl -s "$B/api/autotag?tag=$1" | python3 -c "$2"; }
chk "weak: review-only tag is listed, hopeless is not" "$(lc braces 'import sys,json;d=json.load(sys.stdin);print(sorted(d["tags"]),d["lowConfidenceTags"],d["lowConfidence"])')" "['braces', 'scenery'] ['scenery'] False"
chk "weak: low-confidence flag and its list" "$(lc scenery 'import sys,json;d=json.load(sys.stdin);print(d["lowConfidence"],len(d["pending"]))')" "True 2"
chk "weak: feedback on a review-only suggestion works" "$(post '{"tag":"scenery","id":"7000000000000000030","decision":"rejected"}')" 200
chk "weak: decided items leave the list" "$(lc scenery 'import sys,json;d=json.load(sys.stdin);print(len(d["pending"]),len(d["rejected"]))')" "1 2"
cp /tmp/at_keep.json $AT/autotag.json
# long silent job: status.json heartbeat is stale, but the worker's own heartbeat.json is fresh -> still online
setst '{"state":"running","phase":"training","detail":"tag 3 of 11"}' 600
chk "heartbeat: stale status alone -> offline" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["workerOnline"])')" False
python3 -c "import json,time;json.dump({'t':time.time()},open('$AT/heartbeat.json','w'))"
chk "heartbeat: fresh heartbeat.json -> online while running" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["workerOnline"],d["state"],d["detail"])')" "True running tag 3 of 11"
python3 -c "import json,time;json.dump({'t':time.time()-300},open('$AT/heartbeat.json','w'))"
chk "heartbeat: old heartbeat.json -> offline" "$(curl -s $B/api/autotag/run | python3 -c 'import sys,json;print(json.load(sys.stdin)["workerOnline"])')" False
rm -f $AT/status.json $AT/run_request.json $AT/schedule.json $AT/heartbeat.json $AT/notify_test.json $AT/frames.json
exit $fail
