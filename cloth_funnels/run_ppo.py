import logging
import pdb
import sys
import pprint

from gymnasium.spaces import Box

sys.path.append('/home/lbyang/workspace/cloth-funnels')
from cloth_funnels.utils.utils import (
    setup_actor,
    setup_dataset,
    setup_critic,
    setup_envs,
    seed_all,
    setup_network,
    get_loader,
    get_dataset_size,
    collect_stats,
    collect_coverage_stats,
    get_img_from_fig,
    step_env,
    visualize_value_pred)
import ray
from time import time, strftime
from copy import copy
import torch
from filelock import FileLock
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb

import hydra
import yaml
import glob
import imageio.v3 as iio
import cv2
import shutil
import h5py
import sys
from cloth_funnels.scripts.episode_visualizer import visualize_episode
from cloth_funnels.utils.utils import flatten_dict, get_logger
from omegaconf import DictConfig, OmegaConf
import pathlib
import wandb
from tianshou.env import (
    ContinuousToDiscrete,
    DummyVectorEnv,
    MultiDiscreteToDiscrete,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
    VectorEnvNormObs,
)

from tianshou.utils import WandbLogger, TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer

import numpy as np
import torch
from torch.distributions import Independent, Normal, Categorical
from torchrl.modules.distributions import MaskedCategorical
from torch.utils.tensorboard import SummaryWriter


from learning.nets import MaximumValueActor
from learning.policy import MaximumValuePPOPolicy
from learning.logger import EnvWandbLogger

if __name__ == '__main__':

    # os.environ['WANDB_SILENT']='true'

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(args: DictConfig):
        assert args.name is not None or args.cont is not None, "Must name run or continue run"
        ray.init(local_mode=args.ray_local_mode)
        seed_all(args.seed)

        # config
        time_str = strftime("%m-%d-%H%M")
        if args.name and not args.resume: # if name is not none or evaluating, create a new directory
            run_name = f"{time_str}-{args.name}"
            args.log = os.path.join(args.log_dir, run_name)
            pathlib.Path(args.log).mkdir(parents=True, exist_ok=True)
        if args.resume and not args.eval:
            # assert args.load is None, "Cannot load and continue at the same time"
            cont = f"{args.log_dir}{args.cont}"
            logging.info(f"[RunPPO] Continuing run {args.cont}")
            all_config = yaml.load(open(f'{cont}/config.yaml', 'r'), Loader=yaml.FullLoader)
            # args = OmegaConf.create(all_config['config'])
            args.cont = cont
            run_name = all_config['wandb']['run_name']
            args.log = all_config['config']['log']

        # logger
        get_logger(f'{args.log}/{time_str}_run.log', verbosity=1)
        wandb_meta = {'run_name': run_name, }
        if args.wandb == 'disabled':
            writer = SummaryWriter(args.log)
            # save_interval: epoch
            logger = TensorboardLogger(writer, train_interval=args.episode_length, update_interval=args.episode_length, save_interval=1)
        else:
            logger = EnvWandbLogger(
                train_interval=args.episode_length,
                update_interval=args.episode_length,
                save_interval=1,
                project="ppo-cloth-funnels",
                config=flatten_dict(OmegaConf.to_container(args), sep="/"),  # type: ignore
                name=run_name,
                mode=args.wandb,
                run_id=all_config['wandb']['run_id'] if args.resume else None)

            writer = SummaryWriter(logger.wandb_run.dir)
            logger.load(writer)
            wandb_meta['run_id'] = logger.wandb_run.id
        if not args.resume:
            all_config = {
                'config': OmegaConf.to_container(args, resolve=True),
                'output_dir': os.getcwd(),
                'wandb': wandb_meta
                }
            yaml.dump(all_config, open(f'{args.log}/config.yaml', 'w'), default_flow_style=False)

        # setup train envs, actor
        device = torch.device('cuda:' + str(args.network_gpu))
        actor, _, setup_sucss_stats = setup_actor(args, device)
        dataset_path, test_dataset_path = setup_dataset(args)
        envs, task_loader = setup_envs(dataset=dataset_path, has_ray=False, **args)
        venv = [
            DummyVectorEnv(envs),
            # RayVectorEnv(envs),
            # SubprocVectorEnv(envs)
        ]
        for train_envs in venv:
            state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
            action_shape = train_envs.action_space[0].nvec
            action_space = train_envs.action_space[0]
        critic = setup_critic(args, device, action_shape)
        # model
        # if torch.cuda.is_available():
        #     actor = DataParallelNet(actor).to(device)
        #     critic = DataParallelNet(critic).to(device)
        actor_critic = ActorCritic(actor, critic)

        #  训练actor_critic初始化
        if not args.eval:
            # Trick ： orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            # 只更新critic
            if args.only_critic:
                for param in actor_critic.actor.parameters():
                    param.requires_grad =False

        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
        dist = MaskedCategorical
        policy = MaximumValuePPOPolicy(
            actor,
            critic,
            optim,
            dist,
            action_scaling=isinstance(action_space, Box),
            action_space=action_space,
            deterministic_eval=True,
            advantage_normalization=args.advantage_normalization,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            reward_normalization=args.reward_normalization,
        )

        def save_best_fn(policy):
            torch.save({"model": policy.state_dict(), "optim": optim.state_dict()},
                       os.path.join(args.log, "best_policy.pth"))

        def stop_fn(mean_rewards):
            # return mean_rewards >= args.reward_threshold
            return False
        def save_checkpoint_fn(epoch, env_step, gradient_step):
            ckpt_path = os.path.join(args.log, "latest_ckpt.pth")
            if epoch % 2 == 1:
                ckpt_path = os.path.join(args.log, f"checkpoint_{epoch}.pth")
                torch.save({"model": policy.state_dict(), "optim": optim.state_dict()},
                        ckpt_path)
            torch.save({"model": policy.state_dict(), "optim": optim.state_dict()},
                       ckpt_path)
            return ckpt_path

        if args.wandb == "disabled":
            test_fn = None
        else:
            def test_fn(epoch, global_step):
                # use ray
                args.num_processes = args.test_processes
                args.episode_length = 10
                args.eval = True
                # args.dump_visualizations = True
                test_dataset_path = f'{args.log}/test_replay_buffer{epoch}.hdf5'
                logging.info(f'TEST Replay Buffer path: {test_dataset_path}')
                envs, test_task_loader = setup_envs(dataset=test_dataset_path, **args)
                observations = ray.get([e.reset.remote() for e in envs])
                observations = [obs for obs, _ in observations]
                remaining_observations = []
                ready_envs = copy(envs)
                dataset_size = get_dataset_size(test_dataset_path)
                init_dataset_size = dataset_size
                script_init_time = time()
                step_iter = 0
                print(f"Testing for epoch ：{epoch}, global_step: {global_step}")
                while (True):
                    with torch.no_grad():
                        logging.debug("[RunSim] Stepping env")
                        ready_envs, observations, remaining_observations = \
                            step_env(
                                all_envs=envs,
                                ready_envs=ready_envs,
                                ready_actions=policy.actor.act(observations),
                                remaining_observations=remaining_observations,
                                deterministic=args.deterministic)
                    if step_iter >= args.test_dataset_size  * args.episode_length:
                        dataset_size = get_dataset_size(test_dataset_path)
                        pph = 3600 * (dataset_size - init_dataset_size) / (time() - script_init_time)
                        print("[RunPPO] Points per hour:", pph)
                        # test end
                        if dataset_size - init_dataset_size >= args.test_dataset_size * args.episode_length :
                            logging.info(f'Start Logging Dataset！ Init Dataset Size:{init_dataset_size}, Delta Dataset Size: {dataset_size - init_dataset_size}')
                            for env in envs:
                                ray.kill(env)
                            ray.kill(test_task_loader)

                            start = time()
                            stats_dict = collect_stats(test_dataset_path, action_primitives=args.action_primitives, num_points=512)

                            # collect_coverage_stats
                            coverage_stats_dict = collect_coverage_stats(test_dataset_path, num_points=512)

                            for key, value in coverage_stats_dict.items():
                                if all(word not in key for word in ['distribution', 'img',
                                                                    'min', 'max', '_steps']):
                                    logging.info(f'\t[{key:<36}]:\t{value:.04f}')
                                if key == 'final_coverage/hard/mean' or key == 'best_coverage/hard/mean':
                                    wandb.log({key: float(value), "epoch": epoch, "global_step": global_step})


                            end = time()
                            logging.info(f"Collecting stats took {end - start} seconds")

                            pph = 3600 * (dataset_size - init_dataset_size) / (time() - script_init_time)
                            logging.info('=' * 18 + f' {dataset_size} points ({pph} p/h) ' + '=' * 18)

                            writer.add_scalar("points_per_hours", pph, global_step=epoch)
                            try:
                                with h5py.File(test_dataset_path, "r") as dataset:
                                    fig, axs, _, _ = visualize_episode(stats_dict['vis_key'], test_dataset_path,
                                                                    steps=args.episode_length, vis_index=(0, 1, 2, 3))
                                    try:
                                        wandb.log({"img_episode_vis": wandb.Image(fig), "epoch": epoch, "global_step": global_step})
                                    except:
                                        logging.info("[RunSim] Could not visualize episode, file unavailable")
                                        pass
                                    del stats_dict['vis_key']
                            except:
                                logging.info("[RunSim] Could not visualize episode, file unavailable")
                                pass

                            start = time()
                            try:
                                for key, value in stats_dict.items():
                                    if 'distribution' in key:
                                        sequence = np.array(value, dtype=np.float32)
                                        data = [[s] for s in sequence]
                                        table = wandb.Table(data=data, columns=["scales"])
                                        wandb.log({f'histogram_{key}': wandb.plot.histogram(table, "scales",
                                                                                            title=f"{key}"),"epoch": epoch, "global_step": global_step})

                                    elif 'img' in key:
                                        value = (np.array(value).astype(np.uint8))[:3, :, :].transpose(1, 2, 0)
                                        wandb.log({key: wandb.Image(value), "epoch": epoch, "global_step": global_step})
                                    else:
                                        wandb.log({key: float(value), "epoch": epoch, "global_step": global_step})
                                try:
                                    videos = glob.glob(f'{args.log}/videos/*/*.mp4')
                                    if len(videos):
                                        select_vid = np.random.choice(videos)
                                        # load video as numpy array
                                        frames = []
                                        for i, frame in enumerate(iio.imiter(select_vid)):
                                            frames.append(cv2.resize(frame, (128, 128)))
                                        frames = np.stack(frames, axis=0)
                                        wandb.log({'video': wandb.Video(frames.transpose(0, 3, 1, 2), fps=24), "epoch": epoch, "global_step": global_step})

                                        if not args.dump_visualizations:
                                            for video_dir in glob.glob(f'{args.log}/videos/*'):
                                                shutil.rmtree(video_dir, ignore_errors=True)

                                    logging.info(f"Logging took {time() - start} seconds")
                                except Exception as e:
                                    logging.info("[Video Could Not Be Uploaded]", e)
                                    pass
                            except Exception as e:
                                logging.info(f"[RunSim] Could not log stats {e}")
                                pass
                            os.remove(test_dataset_path)
                            os.remove(test_dataset_path+'.lock')
                            break

                    step_iter += 1

        if args.resume or args.eval or setup_sucss_stats == False:
            # load from existing checkpoint

            if os.path.exists(f'{args.cont}/latest_ckpt.pth'):
                ckpt_path = f'{args.cont}/latest_ckpt.pth'
            elif args.load is not None:
                ckpt_path = os.path.join(args.load)
            if os.path.exists(ckpt_path):
                logging.info(f"Loading agent under {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device)
                try:
                    policy.load_state_dict(checkpoint["model"])
                    optim.load_state_dict(checkpoint["optim"])
                    logging.info("Successfully restore policy and optim.")
                except Exception as e:
                    policy.load_state_dict(checkpoint["model"])
                    logging.info("Successfully restore policy.")
            else:
                logging.info(f"Fail to restore policy and optim: {ckpt_path}")
                exit()

        if args.eval:
            # eval with tianshou
            policy.training = False
            # test_collector = Collector(policy, train_envs)
            # test_collector.collect(n_episode=203)

            # eval with ray
            envs, test_task_loader = setup_envs(dataset=dataset_path, **args)
            observations = ray.get([e.reset.remote() for e in envs])
            observations = [obs for obs, _ in observations]
            remaining_observations = []
            ready_envs = copy(envs)
            while (True):
                with torch.no_grad():
                    logging.debug("[RunSim] Stepping env")
                    ready_envs, observations, remaining_observations = \
                        step_env(
                            all_envs=envs,
                            ready_envs=ready_envs,
                            ready_actions=policy.actor.act(observations),
                            remaining_observations=remaining_observations,
                            deterministic=args.deterministic)
                dataset_size = get_dataset_size(dataset_path)
                if dataset_size >= args.test_dataset_size * args.episode_length:
                    exit(0)

        # train
        else:
            # collector
            train_collector = Collector(
                policy,
                train_envs,
                VectorReplayBuffer(128, len(train_envs)),
            )

            # test env
            args.num_processes = args.test_processes
            args.episode_length = 10
            args.eval = True
            # test_envs, task_loader = setup_envs(dataset=test_dataset_path, has_ray=False, **args)
            # test_env = RLEnv(replay_buffer_path=test_dataset_path,
            #                 get_task_fn=lambda: ray.get(task_loader.get_next_task.remote()),
            #                 recreate_verts=None,
            #                 **args)
            # test_venv = DummyVectorEnv(test_envs)
            # test_collector = Collector(policy, test_venv)

            if args.test_in_train:
                test_collector = Collector(policy, train_envs)
            else:
                test_collector = None

            result = OnpolicyTrainer(
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                max_epoch=args.max_epoch,
                step_per_epoch=args.step_per_epoch,
                repeat_per_collect=args.repeat_per_collect,
                episode_per_test=args.episode_per_test,
                batch_size=args.batch_size,
                episode_per_collect=args.warmup,
                test_fn=test_fn,
                stop_fn=stop_fn,
                save_best_fn=save_best_fn,
                logger=logger,
                save_checkpoint_fn=save_checkpoint_fn,
                test_in_train=args.test_in_train,
                resume_from_log=args.resume,
            ).run()

            pprint.pprint(result)





    main()