from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines import PPO2

from sonic_util import make_sonic_env

def main():
    env = make_sonic_env(game='SonicAndKnuckles3-Genesis', state='FlyingBatteryZone.Act2', stack=True)
    env = DummyVecEnv([lambda: env])

    model = PPO2.load("./models/FlyingBatteryZone.Act2")

    obs = env.reset()
    
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
