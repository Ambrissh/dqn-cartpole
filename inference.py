import torch
import gymnasium as gym
from itertools import count
from model import DQN

class ModelTester:
    """Test trained DQN models"""
    
    def __init__(self, checkpoint_path, n_observations=4, n_actions=2, device=None):
        """
        Initialize tester with a trained model
        
        Args:
            checkpoint_path: Path to checkpoint file
            n_observations: State space dimension (4 for CartPole)
            n_actions: Action space dimension (2 for CartPole)
            device: Device to run on (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        self.model = DQN(n_observations, n_actions).to(self.device)
        self.model.load_state_dict(checkpoint['policy_net'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Trained for {checkpoint['episode']} episodes")
        print(f"Running on {self.device}")
    
    def test(self, env_name='CartPole-v1', num_episodes=10, render=False, verbose=True):
        """
        Test the model
        
        Args:
            env_name: Gym environment name
            num_episodes: Number of episodes to test
            render: Whether to render the environment
            verbose: Whether to print episode results
        
        Returns:
            List of episode rewards
        """
        render_mode = 'human' if render else None
        env = gym.make(env_name, render_mode=render_mode)
        
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            episode_reward = 0
            
            for t in count():
                # Select action (greedy, no exploration)
                with torch.no_grad():
                    action = self.model(state).max(1).indices.view(1, 1)
                
                # Take action
                observation, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                done = terminated or truncated
                
                if done:
                    total_rewards.append(episode_reward)
                    if verbose:
                        print(f"Episode {episode + 1}/{num_episodes}: "
                              f"Reward = {episode_reward:.1f}, Steps = {t + 1}")
                    break
                
                # Move to next state
                state = torch.tensor(observation, dtype=torch.float32, 
                                   device=self.device).unsqueeze(0)
        
        avg_reward = sum(total_rewards) / len(total_rewards)
        if verbose:
            print(f"\n{'='*50}")
            print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
            print(f"Min: {min(total_rewards):.1f}, Max: {max(total_rewards):.1f}")
            print(f"{'='*50}")
        
        env.close()
        return total_rewards


def quick_test(checkpoint_path='checkpoints/latest.pth', episodes=10):
    """Quick test function for convenience"""
    tester = ModelTester(checkpoint_path)
    return tester.test(num_episodes=episodes, render=False)


def demo(checkpoint_path='checkpoints/latest.pth', episodes=3):
    """Visual demo of trained agent"""
    tester = ModelTester(checkpoint_path)
    print("\nStarting visual demo...")
    print("Close the window to continue to next episode\n")
    return tester.test(num_episodes=episodes, render=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained DQN model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    
    args = parser.parse_args()
    
    tester = ModelTester(args.checkpoint)
    tester.test(num_episodes=args.episodes, render=args.render)
