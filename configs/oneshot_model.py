# Experiment
dataset_name                 = '4DCT'            # Dataset name
experiment_name              = 'ontshot_model'
network_type                 = 'ResUNet'         # 'UNet' | 'ResUNet'


# Run instructions
train_model                  = True              # Train the model
valid_model                  = True              # Do validation during training
test_model                   = True              # Test/predict the pre-trained model
manual_seed                  = 0                 # Manual random seed for code [|None]


# Dataset
train_dir                    = 'MC4_SC_cnn_mha'               # Directory of training dataset
valid_dir                    = 'MC4_SC_cnn_mha'           # Directory of validation dataset
test_dir                     = 'MC4_SC_cnn_mha'          # Directory of test dataset
data_suffix                  = '.mha'
phase_num                    = 10
volume_size                  = (224, 96, 224)           # Size of npy file
# ITK image setting
direction                    = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
train_spacing                = (2, 2, 2)
valid_spacing                = (2, 2, 2)
test_spacing                 = (2, 2, 2)
val_landmark_evl             = False                 # Registration error based on validation landmark
test_landmark_evl            = False                 # Registration error based on testing landmark


# Network parameters
reg_model                    = 'Group'           # ['Pair' | 'Group']
dvf_scale                    = 1

# Training parameters
optimizer                    = 'adam'                   # Optimizer ['sgd' | 'adam' | 'nadam']
learning_rate_base           = 1e-3                     # Base learning rate
n_epochs                     = 2000                     # Number of epochs during training
batch_size                   = 1
L2_penalty                   = 0                        # L2 penalty for regularization
load_weights_file            = 'best_model.pth'      # Weights file to be loaded, ['final_weights.pth' | 'best_model.pth']

# Loss
window_size                  = 5                  # Window size for local NCC or SSIM
simi_loss_type               = 'NCC2'             # ['NCC' |'NCC2'| 'SSIM' | 'MSE']
smooth_loss_type             = 'dvf'              # ['dvf' | 'dvf_image' | None]
difference_type              = 'forward'          # ['forward' | 'central']
s1w                          = 0                  # Weight for 1st-order spatial regularization    
s2w                          = 1               # Weight for 2nd-order spatial regularization     
t1w                          = 1e-5               # Weight for 1st-order temporal regularization  
t2w                          = 0                  # Weight for 1st-order temporal regularization  
cyclic_reg                   = 5e-2


# For Unet
convs_per_level              = 2
enc_filters                  = [32, 64, 128, 256]      # [32, 64, 64, 64], length = unet level
dec_filters                  = [128, 64, 32]           # [64, 64, 64, 32, 32]
activation                   = 'LeakyReLU'
normalization                = 'Instance'           # None | 'Instance'
downsampling                 = 'MaxPool'            # ['MaxPool' | 'StrideConv' | '']
upsampling                   = 'UpSample'           # ['UpSample' | 'TransConv' | '']

# For STN
STN_interpolation            = 'bilinear'


# Test parameters
target_indexes               = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]               # Targeted phase, all phase to this targeted phase
save_input_itk               = True              # Save input phase images
save_dvf2phase               = True              # Save DVF deforming phase images to the target
save_deformed_img            = True              # Save all phase images deformed to the target
save_template_related        = True              # Save the template with corresponding DVFs and deformed images


# Callback learning rate scheduler
LRScheduler_type             = 'no_decay'          # Type of scheduler ['no_decay' | 'exp_decay' | 'power_decay' | 'step_decay' | ...]
LRScheduler_gamma            = 0.9                 # Power for the power_decay, exp_decay modes
LRScheduler_arg              = 20                  # Additional argument of scheduler, step for step_decay, ratio for boundary_decay
    
# Check point saving
checkpoint_enabled           = True            # Enable the Callback
checkpoint_save_freq_epoch   = 20              # Frequency in epoch

# Stop criterion
earlyStopping_enabled        = True           # Enable the Stop criterion
stop_type                    = 'converge_loss' # ['converge_std' | 'converge_loss' | 'improve']
stop_threshold               = 0.00001         # Threshold for the std or minimum differenece with previous, or minimum loss improvement
stop_query_len               = 20              # Length for std or loss calculation
stop_min_iter                = 20              # Min iter numbers for stopping
save_best_model              = True            # Save the best model or last model