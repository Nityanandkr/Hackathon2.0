import re

filepath = 'c:/Users/nitya/Downloads/Hackathon2.0/scripts/train_rf_model.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update FEATURE_ORDER
new_features = """FEATURE_ORDER = [
    "mouse_speed_avg", "mouse_speed_variance", "mouse_direction_changes",
    "mouse_curve_variance", "mouse_idle_ratio", "typing_interval_avg",
    "typing_interval_variance", "typing_burst_count", "backspace_frequency",
    "scroll_event_count", "scroll_speed_avg", "click_frequency",
    "session_duration_ms", "browser_entropy",
    "device_is_mobile", "device_memory_gb", "hardware_concurrency",
    "js_challenge_time_ms", "js_challenge_success", "js_challenge_score"
]"""
content = re.sub(r'FEATURE_ORDER\s*=\s*\[.*?\]', new_features, content, flags=re.DOTALL)

def add_fields(func_name, fields):
    global content
    pattern = rf'(def {func_name}\(\):.*?return \[.*?)(    \])'
    content = re.sub(pattern, rf'\1{fields}\2', content, flags=re.DOTALL)

human_additions = {
    '_human_casual_browser': '        0.0,                                        # device_is_mobile\n        random.choice([4.0, 8.0, 16.0]),            # device_memory_gb\n        random.choice([4.0, 8.0, 16.0]),            # hardware_concurrency\n        random.uniform(30.0, 100.0),                # js_challenge_time_ms\n        1.0,                                        # js_challenge_success\n        random.uniform(800, 1200),                  # js_challenge_score\n',
    '_human_fast_typer': '        0.0,\n        random.choice([8.0, 16.0, 32.0]),\n        random.choice([8.0, 12.0, 16.0]),\n        random.uniform(15.0, 50.0),\n        1.0,\n        random.uniform(1000, 1500),\n',
    '_human_mobile_user': '        1.0,\n        random.choice([3.0, 4.0, 6.0, 8.0]),\n        random.choice([4.0, 6.0, 8.0]),\n        random.uniform(80.0, 200.0),\n        1.0,\n        random.uniform(400, 800),\n',
    '_human_elderly_user': '        0.0,\n        random.choice([4.0, 8.0]),\n        random.choice([4.0, 8.0]),\n        random.uniform(50.0, 150.0),\n        1.0,\n        random.uniform(600, 1000),\n',
    '_human_gamer': '        0.0,\n        random.choice([16.0, 32.0, 64.0]),\n        random.choice([12.0, 16.0, 24.0]),\n        random.uniform(10.0, 30.0),\n        1.0,\n        random.uniform(1500, 2500),\n',
    '_human_form_filler': '        0.0,\n        random.choice([8.0, 16.0]),\n        random.choice([8.0, 16.0]),\n        random.uniform(20.0, 80.0),\n        1.0,\n        random.uniform(800, 1300),\n'
}

bot_additions = {
    '_bot_instant': '        0.0,\n        0.0,\n        0.0,\n        0.0,\n        0.0,\n        0.0,\n',
    '_bot_linear': '        0.0,\n        0.0,\n        0.0,\n        random.uniform(0.0, 15.0),\n        0.0,\n        0.0,\n',
    '_bot_replay': '        0.0,\n        8.0,\n        4.0,\n        random.uniform(5.0, 20.0),\n        random.choice([0.0, 1.0]),\n        random.uniform(0.0, 300.0),\n',
    '_bot_headless_browser': '        0.0,\n        0.0,\n        0.0,\n        random.uniform(5.0, 40.0),\n        random.choice([0.0, 1.0]),\n        random.uniform(0.0, 500.0),\n',
    '_bot_credential_stuffer': '        0.0,\n        2.0,\n        1.0,\n        random.uniform(0.0, 10.0),\n        0.0,\n        0.0,\n',
    '_bot_slow_scraper': '        0.0,\n        4.0,\n        2.0,\n        random.uniform(10.0, 100.0),\n        1.0,\n        random.uniform(100.0, 600.0),\n',
    '_bot_click_farmer': '        random.choice([0.0, 1.0]),\n        random.choice([2.0, 4.0]),\n        random.choice([2.0, 4.0]),\n        random.uniform(50.0, 300.0),\n        1.0,\n        random.uniform(200.0, 800.0),\n',
    '_bot_api_abuser': '        0.0,\n        0.0,\n        0.0,\n        0.0,\n        0.0,\n        0.0,\n'
}

for f_name, repl in human_additions.items():
    add_fields(f_name, repl)

for f_name, repl in bot_additions.items():
    add_fields(f_name, repl)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated script successfully.')
