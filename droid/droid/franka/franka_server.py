import numpy as np
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from franka_robot import FrankaRobot


app = FastAPI()

robot: Optional[FrankaRobot] = None


@app.on_event("startup")
def startup():
    global robot
    robot = FrankaRobot()


class UpdateCommandRequest(BaseModel):
    command: List[float]
    action_space: str = "cartesian_velocity"
    gripper_action_space: Optional[str] = None
    blocking: bool = False


class UpdatePoseRequest(BaseModel):
    command: List[float]
    velocity: bool = False
    blocking: bool = False


class UpdateJointsRequest(BaseModel):
    command: List[float]
    velocity: bool = False
    blocking: bool = False
    cartesian_noise: Optional[List[float]] = None


class UpdateGripperRequest(BaseModel):
    command: float
    velocity: bool = True
    blocking: bool = False


class CreateActionDictRequest(BaseModel):
    action: List[float]
    action_space: str
    gripper_action_space: Optional[str] = None
    robot_state: Optional[Dict[str, Any]] = None


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()

    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().tolist()

    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}

    if isinstance(x, list):
        return [to_jsonable(v) for v in x]

    if isinstance(x, tuple):
        return tuple(to_jsonable(v) for v in x)

    if isinstance(x, np.generic):
        return x.item()

    return x


@app.post("/update_command")
def update_command(req: UpdateCommandRequest):
    result = robot.update_command(
        command=np.array(req.command),
        action_space=req.action_space,
        gripper_action_space=req.gripper_action_space,
        blocking=req.blocking,
    )
    return {"result": to_jsonable(result)}


@app.post("/update_joints")
def update_joints(req: UpdateJointsRequest):
    cartesian_noise = None
    if req.cartesian_noise is not None:
        cartesian_noise = np.array(req.cartesian_noise)

    result = robot.update_joints(
        command=np.array(req.command),
        velocity=req.velocity,
        blocking=req.blocking,
        cartesian_noise=cartesian_noise,
    )
    return {"result": to_jsonable(result)}


@app.post("/update_gripper")
def update_gripper(req: UpdateGripperRequest):
    result = robot.update_gripper(
        command=req.command,
        velocity=req.velocity,
        blocking=req.blocking,
    )
    return {"result": to_jsonable(result)}


@app.get("/get_robot_state")
def get_robot_state():
    state_dict, timestamp_dict = robot.get_robot_state()
    return {
        "state_dict": to_jsonable(state_dict),
        "timestamp_dict": to_jsonable(timestamp_dict),
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8006,
    )
