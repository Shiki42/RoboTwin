#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from checkpoint_assets import checkpoint_asset_id
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config

logger = logging.getLogger(__name__)


def create_policy(train_config_name: str, checkpoint_dir: Path):
    asset_id = checkpoint_asset_id(checkpoint_dir / "assets")
    return policy_config.create_trained_policy(
        config.get_config(train_config_name),
        checkpoint_dir,
        robotwin_repo_id=asset_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = create_policy(args.train_config_name, args.checkpoint_dir)
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    logger.info(
        "Serving %s from %s on %s:%d",
        args.train_config_name,
        args.checkpoint_dir,
        args.host,
        args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
