import gymnasium as gym   # ✅ cambiato da gym → gymnasium
import np as np

class VideoWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            enabled=True,
            steps_per_render=1,
            **kwargs
        ):
        super().__init__(env)
        
        self.mode = mode
        self.enabled = enabled
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render

        self.frames = list()
        self.step_count = 0

    def reset(self, **kwargs):
        # ✅ Gymnasium reset() → (obs, info)
        obs, info = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        if self.enabled:
            # ✅ Gymnasium usa render_mode definito a creazione env
            frame = self.env.render(**self.render_kwargs)
            assert frame.dtype == np.uint8
            self.frames.append(frame)
        return obs, info   # ✅ ritorna anche info (Gymnasium API)
    
    def step(self, action):
        # ✅ Gymnasium step() → (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1
        if self.enabled and ((self.step_count % self.steps_per_render) == 0):
            frame = self.env.render(**self.render_kwargs)  # ✅ niente argomento mode
            assert frame.dtype == np.uint8
            self.frames.append(frame)
        # ✅ restituisce i 5 valori come da Gymnasium
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array', **kwargs):
        return self.frames
