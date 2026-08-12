"""Static checks for the maintained alohamini_sim AlohaMini2Pro asset."""

import hashlib
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "alohamini_sim/data_engine/agents/aloha_mini/assets/alohamini2pro"
SYNC_SCRIPT = ROOT / "alohamini_sim/scripts/sync_alohamini2pro_asset.py"


def test_asset_tree_mass_and_dof():
    root = ET.parse(ASSET / "urdf/alohamini2pro.urdf").getroot()
    links = root.findall("link")
    active = [joint for joint in root.findall("joint") if joint.get("type") != "fixed"]
    masses = [float(link.find("inertial/mass").get("value")) for link in links]
    assert len(links) == 29
    assert len(active) == 18
    assert np.isclose(sum(masses), 16.0)


def test_installed_sts3095_c002_parameters_are_present():
    config_path = ASSET / "config/actuator_parameters.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    servo = config["servos"]["sts3095_c002"]
    assert np.isclose(servo["simulation_velocity_limit_rad_s"], 2.48)
    assert np.isclose(servo["simulation_force_limit_nm"], 2.21)
    lift = config["joint_mapping"]["lift"]
    assert np.isclose(lift["simulation_velocity_limit_m_s"], 0.05)
    assert np.isclose(lift["simulation_force_limit_n"], 74.2)

    source = config["source"]["installed_variant_overrides"]["sts3095_c002"]
    snapshot = (config_path.parent / source["path"]).resolve()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == source["sha256"]


def test_asset_matches_canonical_source_when_available():
    spec = importlib.util.spec_from_file_location("sync_alohamini2pro_asset", SYNC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not module.DEFAULT_SOURCE.is_dir():
        return
    missing, different = module.compare(module.DEFAULT_SOURCE, module.DEFAULT_DESTINATION)
    assert missing == []
    assert different == []
