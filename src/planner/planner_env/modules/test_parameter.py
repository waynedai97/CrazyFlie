#Environment parameters
MAXIMUM_AGENTS = 5                 # Define max number of agents in the environment
NUM_HEADING = 4                    # Number of heading for each node
SPEED = 1                          # Speed of the agent in metres per second
SENSOR_RANGE = 70                  # Sensor range in pixels
SCALE = 32                         # Scale of the environment (pixels per meter)
COVERAGE = False                   # Coverage task

# Model parameters
REPLAY_SIZE = 15000                # Replay buffer size in terms of training steps, depends on RAM
MINIMUM_BUFFER_SIZE = 2000         # Minimum buffer size before starting training
BATCH_SIZE = 128 #8                # Batch size for training, depends on GPU 11gb for 64
if COVERAGE:                       # Add coverage observations to the input
    add = 1
else:
    add = 0
INPUT_DIM = 8 + add                # Change to 8 for multi-agent exploration with heading
LR = 1e-5                          # For Adam optimizer
GAMMA = 1
DECAY_STEP = 256                   # Decay step for learning rate scheduler
MAXIMUM_TIMESTEPS = 128

# Graph parameters
ADAPTIVE_GENERATION = False        # Generate nodes based on the speed
SHOW_HEADINGS = True               # Show heading in the graph
EMBEDDING_DIM = 128                # Embedding dimension for node
NODE_PADDING_SIZE = 360            # Total number of nodes in the graph
K_SIZE = 20                        # Number of neighbors
RESOLUTION = 4                     # Downsampling resolution for the graph

# GPU parameters
USE_GPU_DATA= True                 # For data collection
USE_GPU_TRAINING = True             # For training
NUM_GPU = 'AUTO'                    # Use 'AUTO' to automatically detect number of GPUs
NUM_META_AGENT = 22 #6

# Data handling
SUMMARY_WINDOW = 32                 # Write to tensorboard
FOLDER_NAME = 'adaptive_heading_11_22_Best'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
LOAD_MODEL = False
SAVE_MODEL_INTERVAL = 32           # Number of episodes between saving model
SAVE_IMG_GAP = 100                 # Number of episodes between saving images
PLOT_EDGE = True                   # Plot edges in the graph

# Display episode summary
SUMMARY = True
SUMMARY_INTERVAL = 35
TIMESTEP_INTERVAL = 32
