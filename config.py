# OMNIKON — Person A Pipeline: Architecture Configuration

BATCH_SIZE       = 32     # Default batch size for training
IMG_SIZE         = 299    # Matches Person B's spatial resolution
RGB_CHANNELS     = 3      # Standard RGB input for the visual stream
PHYSICS_CHANNELS = 2      # Defocus Map + Specular Map for the physics stream
LATENT_DIM       = 512    # Target dimension for both feature streams
DROPOUT_RATE     = 0.3    # Network regularization to prevent overfitting