import os
import pathlib
from os.path import abspath, dirname
from dynaconf import Dynaconf

current_dir = dirname(abspath(__file__))

# 1. Package defaults
package_toml_files = list(
    pathlib.Path(current_dir).glob("*.toml")
)

toml_files = package_toml_files.copy()

# 2. External configuration
config_path = os.getenv("ALPHA_CODIUM_CONFIG_FILE")
if config_path and pathlib.Path(config_path).exists():
    toml_files.append(config_path)

# 3. External secrets
secrets_path = os.getenv("ALPHA_CODIUM_SECRETS_FILE")
if secrets_path and pathlib.Path(secrets_path).exists():
    toml_files.append(secrets_path)

global_settings = Dynaconf(
    envvar_prefix=False,
    merge_enabled=True,
    settings_files=toml_files,
)

def get_settings():
    return global_settings
