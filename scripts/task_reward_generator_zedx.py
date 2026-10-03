import os
import sys

import cv2
import torch
import time, threading, queue, random
import subprocess
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime
from pathlib import Path

import pyzed.sl as sl
import base64
from dotenv import load_dotenv
from openai import OpenAI
from groundingdino.util.inference import load_model, load_image, predict, annotate
from loguru import logger

from fastapi import FastAPI
import uvicorn
import zerorpc
import asyncio
import websockets
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json_numpy
json_numpy.patch()

from zedx_streamer import ZEDStreamer


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))

class GPTServer:
    def __init__(self):
        self.task_reward_generator = TaskRewardGenerator(
            n_tasks=2,
            max_timesteps=60,
            max_rounds=1, # max_rounds=n means the robot performs n+1 rounds
            zed_stream_ip="192.168.55.1",
            zed_stream_port=30002,  # zed_left: 30000 / zed_head: 30002 / zed_right: 30004
        )
        self.round = 0
        self.scene_before = None
        self.scene_after = None
        self.objects_all = None
        self.selected_task = None
        self.img_rgb = None

    def run(self, port = 8000, host = "0.0.0.0"):
        self.app = FastAPI()
        self.app.post("/query_task")(self.task_generation)
        self.app.post("/query_reward")(self.reward_generation)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset(self):
        pass


    # task_0-(reward_0-task_1)-(reward_1-task_2)-...
    def task_generation(self, payload: Dict[Any, Any]):
        img_rgb = payload["img"]  # shape: (1080, 1920, 3) dtype: uint8
        if img_rgb is not None:  # round=0
            scene_dino, scene_gpt = self.task_reward_generator.process_img(img_rgb)

            self.objects_all = self.task_reward_generator.objects_generation(scene_gpt)
            tasks, objects = self.task_reward_generator.task_generation(scene_gpt, self.objects_all)

            self.scene_before = self.task_reward_generator.gdino(self.objects_all, scene_dino, self.round)
        else:  # round=1,2,...
            scene_dino, scene_gpt = self.task_reward_generator.process_img(self.img_rgb)

            self.objects_all = self.task_reward_generator.objects_generation(scene_gpt)
            tasks, objects = self.task_reward_generator.task_generation(scene_gpt, self.objects_all)


        logger.info(f"Objects in the scene: \n{self.objects_all}")
        logger.info("Tasks in the scene:")
        for task in tasks:
            logger.info(f"  {task}")
        logger.info("Objects of each task:")
        for obj in objects:
            logger.info(f"  {obj}")

        self.selected_task = self.task_reward_generator.task_selection(tasks, objects)
        self.round += 1

        return {
            "task": self.selected_task,
            "round": self.round,
        }

    def reward_generation(self, payload: Dict[Any, Any]):
        img_rgb = payload["img"]

        next_scene_dino, next_scene_gpt = self.task_reward_generator.process_img(img_rgb)
        self.scene_after = self.task_reward_generator.gdino(self.objects_all, next_scene_dino, self.round)

        reward = self.task_reward_generator.reward_generation(self.scene_before, self.scene_after, self.selected_task)

        self.img_rgb = img_rgb
        self.scene_before = self.scene_after

        return {
            "reward": reward
        }

class TaskRewardGenerator:
    def __init__(
        self, 
        n_tasks: int = 2, 
        max_timesteps: int = 60, 
        max_rounds: int = 1, 
        zed_stream_ip: str = "192.168.55.1",
        zed_stream_port: int = 30000,
    ):
        self.n_tasks = n_tasks
        self.max_timesteps = max_timesteps
        self.current_scene_gdino = None
        self.next_scene_gdino = None

        self.zed = None
        self.camera = self.camera = ZEDStreamer()
        self.camera.start(stream_ip=zed_stream_ip, stream_port=zed_stream_port)
        self.image = None

        self.round_current = 0
        self.round_next = 1
        self.max_rounds = max_rounds
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.queue: "Queue[tuple[list[str], list[str]]]" = Queue()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "GroundingDINO/weights/groundingdino_swint_ogc.pth",
        ).to(self.device).eval()
        self.runtime_parameters = None

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.setup_logger()
        logger.info("TaskRewardGeneration initialised (tasks: {}, max_timesteps: {}s)", n_tasks, max_timesteps)

    def setup_logger(self):
        log_dir = f"outputs/task_reward_generation/{self.timestamp}"
        log_path = os.path.join(log_dir, "log.txt")

        logger.remove()
        logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    def mask_image(self, image):
        """Keep the tabletop polygon and fill the remaining pixels with the background color."""
        polygon_points = [
            [1500, 150],
            [1600, 545],
            [1310, 535],
            [1320, 715],
            [1280, 715],
            [1285, 830],

            [600, 820],
            [615, 715],
            [575, 715],
            [605, 530],
            [345, 520],
            [460, 135],
        ]
        background_color=(0, 0, 0)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygon = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)

        masked_image = np.full_like(image, background_color, dtype=image.dtype)
        masked_image[mask > 0] = image[mask > 0]

        return masked_image

    def process_img(self, img_rgb):
        img_rgb = np.asarray(img_rgb, dtype=np.uint8)
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(
                f"Expected an HWC RGB image with 3 channels, got shape={img_rgb.shape}"
            )

        # OpenCV's JPEG encoder expects BGR, while the API boundary uses RGB.
        # Convert exactly once here so the encoded image sent to GPT keeps the
        # original colors.
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


        ok, buffer = cv2.imencode(".jpg", img_bgr)
        if ok:
            b64jpg = base64.b64encode(buffer.tobytes()).decode("utf-8")
        else:
            raise RuntimeError("Failed to encode image to JPEG")

        masked_img_bgr = self.mask_image(img_bgr)

        return masked_img_bgr, b64jpg

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def gdino(self, object, scene, round):
        TEXT_PROMPT = object
        BOX_TRESHOLD = 0.32
        TEXT_TRESHOLD = 0.25

        image_source, image = load_image(scene)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        save_dir = f"outputs/task_reward_generation/{self.timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"annotated_image_{round}.jpg")
        cv2.imwrite(save_path, annotated_frame)

        scene_gdino = self.encode_image(save_path)

        return scene_gdino

    def objects_generation(self, scene):
        prompt_objects = (
            "The table is covered with a black tablecloth, list all the main objects on it.\n"
            "The answer do **not** include:\n"
            "- the tablecloth itself.\n"
            "- the robot itself.\n"
            "- any adjectives, such as color, size, material, or quantity.\n"

            "Format your answer as a single line, separating each object with ' . ' and ending with a final ' .'\n"
            "Correct example: 'cube . tomato . banana . carrot . spoon . cloth . bowl . can .'\n"
            "Incorrect example: 'blue cube . small tomato . object . soup spoon . plastic bowl.'\n"
        )

        response = self.client.responses.create(
            model="gpt-5.6-terra",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_objects},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{scene}", "detail": "high"},
                ],
            }],
        )

        objects_all = response.output_text

        return objects_all

    def task_generation(self, scene, objects_all):
        banned_phrases = [
            "next to", "next", "nearby", "near", "parallel", "close to",
            "on the edge of", "edge", "corner", "according to their orientation",
            "adjacent to", "beside", "alongside", 
            "sides of the table", "separate sides", "both sides"
        ]

        valid_tasks_dict = {}   # taskN: line
        valid_objects_dict = {} # taskN: line
        attempt = 0
        max_attempts = 5
        expected_ids = [f"task{i+1}" for i in range(self.n_tasks)]

        while len(valid_tasks_dict) < self.n_tasks and attempt < max_attempts:
            attempt += 1
            remaining = self.n_tasks - len(valid_tasks_dict)

            prompt_task = (
                f"You are a robot with one arm. Based on the scene, generate {remaining} possible tasks the robot could perform.\n"
                f"ONLY use objects from this EXACT list (no more, no less):\n{objects_all}\n"
                "You are NOT allowed to invent new objects.\n"
                "If any object appears in the task that is not in the list, the task is INVALID.\n"
                "Your tasks must follow the allowed action types below, and you MUST replace 'object' and 'container' with specific object names:\n"
                "- Pick up an object from one place and place it into a container.\n"
                "- pick up an object in a container and place it in front of another object.\n"
                "- pick up an object in a container and place it behind another object.\n"
                "- pick up an object in a container and place it to the left of another object.\n"
                "- pick up an object in a container and place it to the right of another object.\n"
                "- pick up an object and place it in front of another object.\n"
                "- pick up an object and place it behind another object.\n"
                "- pick up an object and place it to the left of another object.\n"
                "- pick up an object and place it to the right of another object.\n"

                "Correct format example:\n"
                "task1: pick up the tomato and place it into the bowl.\n"
                "task2: pick up the banana and place it into the plate.\n"
                "task3: pick up the cube in the bowl and place it in front of the bowl.\n"
                "task4: pick up the carrot and place it in front of the can.\n"


                "Now, for each task above, extract the objects involved.\n"
                "Follow these **strict rules**:\n"
                "- You must write exactly one line per task.\n"
                "- Each line must begin with `taskN:` (e.g. `task1:`).\n"
                "- After the colon, list all object names (nouns only), separated by ` . `, and end with a final ` .`\n"
                "- Do **not** include any adjectives (e.g. color, size, quantity).\n"
                "- Do **not** merge the objects from multiple tasks into a single list.\n"
                "- The word 'side', 'task' is not considered an object.\n\n"

                "Correct format example:\n"
                "objects:\n"
                "task1: tomato . bowl .\n"
                "task2: banana . plate .\n"
                "task3: cube . bowl .\n"
                "task4: carrot . can .\n"
            )   

            response = self.client.responses.create(
                model="gpt-5.6-terra",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_task},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{scene}", "detail": "high"},
                    ],
                }],
            )

            output = response.output_text
            if "objects:" in output:
                task_block, object_block = output.split("objects:", 1)
            else:
                task_block, object_block = output, ""

            task_lines = [line.strip() for line in task_block.splitlines() if line.lower().startswith("task")]
            object_lines = [line.strip() for line in object_block.splitlines() if line.lower().startswith("task")]

            task_raw = {line.split(":")[0].strip(): line for line in task_lines}
            object_raw = {line.split(":")[0].strip(): line for line in object_lines}

            missing_ids = [tid for tid in expected_ids if tid not in valid_tasks_dict]
            missing_ids_iter = iter(missing_ids)

            for old_id, task_line in task_raw.items():
                task_body = task_line.split(":", 1)[1].strip()

                if any(phrase in task_body.lower() for phrase in banned_phrases):
                    continue

                try:
                    new_task_id = next(missing_ids_iter)
                except StopIteration:
                    break

                valid_tasks_dict[new_task_id] = f"{new_task_id}: {task_body}"

                if old_id in object_raw:
                    obj_body = object_raw[old_id].split(":", 1)[1].strip()
                    valid_objects_dict[new_task_id] = f"{new_task_id}: {obj_body}"

        if len(valid_tasks_dict) < self.n_tasks:
            logger.warning(f"Only {len(valid_tasks_dict)} valid tasks generated after {attempt} attempts (target: {self.n_tasks}).")

        sorted_ids = sorted(valid_tasks_dict.keys(), key=lambda x: int(x[4:]))
        task_output = "\n".join([valid_tasks_dict[tid] for tid in sorted_ids])
        object_output = "\n".join([valid_objects_dict[tid] for tid in sorted_ids if tid in valid_objects_dict])

        tasks = [line.strip().strip() for line in task_output.splitlines() if line.strip().startswith("task")]
        objects = [line.strip().strip() for line in object_output.splitlines() if line.strip().startswith("task")]

        return tasks, objects

    def reward_generation(self, current_scene_gdino, next_scene_gdino, current_task):   
        prompt_reward = (
            "You are given two images:\n"
            "- The **first image** shows the initial scene **before** the robot starts the task.\n"
            "- The **second image** shows the result **after** the robot attempted the task.\n\n"

            "The robot was instructed to perform the following task:\n"
            f"{current_task}\n\n"

            "Instructions:\n"
            "1. From the camera's perspective, carefully look at the **initial position** of the key object(s) mentioned in the task.\n"
            "   - The top of the image represents the **front** of the object.\n"
            "   - The bottom of the image represents the **behind** of the object.\n"
            "   - The left side of the image represents the **left side** of the object.\n"
            "   - The right side of the image represents the **right side** of the object.\n"
            "2. Pay attention to whether the object is inside something like a bowl or container.\n"
            "3. Then carefully look at the **final position** of the key object in the second image.\n"
            "4. **VERY IMPORTANT: If bounding boxes are visible, focus on the exact **box labels** — make sure you refer to the correct object name!**\n"
            "   - Do NOT confuse objects with similar color/shape.\n"
            "   - If labels clearly show containment or alignment, include that in your reasoning.\n"
            "5. Compare the final position with the task requirement. "

            "**Respond with only a single digit: `1` if the task was successfully completed, or `0` if it failed.**\n"
            "**Do not give a reason**, just output a single digit:\n"
            "If the key object(s) are placed exactly as instructed, output `1`.\n"
            "If not (wrong position, ambiguous, or missing), output `0`.\n\n"
        )   

        response = self.client.responses.create(
            model="gpt-5.6-terra",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_reward},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{current_scene_gdino}", "detail": "high"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{next_scene_gdino}", "detail": "high"},
                ],
            }],
        )

        reward = response.output_text

        return reward

    def task_selection(self, tasks, objects):


        task_id = 0
        task = "pick up the tomato and place it into the bowl"
        logger.info(f"Selected {task_id} to execute: {task}")

        return task


if __name__ == '__main__':
    gpt_server = GPTServer()
    gpt_server.run(host="0.0.0.0", port=8007)
