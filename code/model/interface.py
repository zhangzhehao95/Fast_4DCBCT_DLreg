import os
import torch
import numpy as np
from scipy import interpolate

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_generator import GroupDataset
import model.losses as losses
from model.networks import RegGroup
from model.modules import ProcessDisp
from utils.lr_schedule import LR_schedule
from utils.helper import save_as_itk, StopCriteriaConvergeStd, StopCriteriaConvergeLoss, StopCriteriaImprove

from torchinfo import summary
import tqdm


# Choose function for similarity calculation
def loss_function(dim, cf, device):
    if cf.simi_loss_type == 'NCC':
        loss_func = losses.NCC_loss(dim=dim, windows_size=cf.window_size).to(device=device)
    elif cf.simi_loss_type == 'NCC2':
        loss_func = losses.NCC2_loss(dim=dim, windows_size=cf.window_size).to(device=device)
    elif cf.simi_loss_type == 'SSIM':
        loss_func = losses.SSIM_loss(spatial_dims=dim, win_size=cf.window_size).to(device=device)
    elif cf.simi_loss_type == 'MSE':
        loss_func = losses.MSE_loss().to(device=device)
    else:
        raise NotImplementedError(cf.simi_loss_type)

    return loss_func


# Registration error based on landmark
def landmark_reg_error(composed_dvf, landmark_path, spacing=[2, 2, 2]):
    """
    Registration error = || ls - lt - DVF(lt) ||, here source is phase 5 and target is phase 0
    composed_dvf: the inter-phase DVFs, (n, 1, 3, d, h, w)
    """
    landmark_file_00 = os.path.join(landmark_path, r'Converted_landmark_T00_xzy.txt')
    landmark_file_50 = os.path.join(landmark_path, r'Converted_landmark_T50_xzy.txt')
    try:
        landmark_00 = np.loadtxt(landmark_file_00, dtype=np.int64)
        landmark_50 = np.loadtxt(landmark_file_50, dtype=np.int64)
    except IOError:
        print(f'Unable to read landmark files.')
        return -1

    # Displacement from phase5 to phase 0
    landmark_disp = landmark_50 - landmark_00   # Original coordinate of landmark follows (x,z,y) order

    pixel_spacing = np.array(spacing, dtype=np.float32)

    dvf50 = composed_dvf[5, 0, ...].cpu().numpy()   # (3(d,h,w), d, h, w)
    dvf50 = np.transpose(np.flip(dvf50, axis=0), (3, 2, 1, 0))  # (w, h, d, 3 (w,h,d))

    # Provide the grids and corresponding values on the grids for interpolator
    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in dvf50.shape[:-1]]
    inter = interpolate.RegularGridInterpolator(grid_tuple, dvf50)

    calc_landmark_disp = inter(landmark_00)     # [300, 3], here 3 indicates 3-dim DVF

    diff = (np.sum(((landmark_disp - calc_landmark_disp) * pixel_spacing) ** 2, 1)) ** 0.5
    print(f'Landmark RE: {np.mean(diff):.2f}+-{np.std(diff):.2f}({np.max(diff):.2f})')
    return np.mean(diff)


def groupwise_train(device, cf):
    # Load training data
    train_dataset = GroupDataset(cf.train_data_path, cf.phase_num)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                              worker_init_fn=np.random.seed(0))
    n_train = len(train_dataset)
    dim = len(cf.volume_size)

    # Load validation data
    if cf.valid_model:
        val_dataset = GroupDataset(cf.val_data_path, cf.phase_num)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                worker_init_fn=np.random.seed(0))
        n_val = len(val_loader)
        # Always show SSIM performance
        if cf.simi_loss_type != 'SSIM':
            ssim_metrics = losses.SSIM_loss(spatial_dims=dim, win_size=cf.window_size).to(device=device)
        else:
            ssim_metrics = None

    # Network
    net = RegGroup(image_shape=cf.volume_size, group_num=cf.phase_num, cf=cf)
    net.to(device=device)
    summary(net, input_size=(1, cf.phase_num, *cf.volume_size))     # (batch, channel, *spatial_dim)

    # Calculate the registration error through landmark using DVF between phase 5 and phase 0.
    if cf.val_landmark_evl:
        process_disp = ProcessDisp(disp_shape=cf.volume_size, calc_device=device)

    # Continue previous training
    if cf.load_pretrained:
        net.load_state_dict(torch.load(cf.pretained_weights_file, map_location=device))
        print("\n > Loading pre-trained weights: " + cf.pretained_weights_file)

    # Loss function
    simi_loss_func = loss_function(dim, cf, device)
    smooth_loss_func = losses.SmoothRegularization(cf.s1w, cf.s2w, cf.t1w, cf.t2w, cf.smooth_loss_type,
                                                   cf.difference_type).to(device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate_base, weight_decay=cf.L2_penalty)
    scheduler = LR_schedule(optimizer, cf.LRScheduler_type, gamma=cf.LRScheduler_gamma, tol_epochs=cf.n_epochs,
                            arg=cf.LRScheduler_arg)

    # Choose stopping criteria, used for one-shot learning
    if cf.stop_type == 'converge_std':
        stop_criterion = StopCriteriaConvergeStd(stop_std=cf.stop_threshold, query_len=cf.stop_query_len,
                                                 num_min_iter=cf.stop_min_iter)
    elif cf.stop_type == 'converge_loss':
        stop_criterion = StopCriteriaConvergeLoss(difference=cf.stop_threshold, query_len=cf.stop_query_len,
                                                  num_min_iter=cf.stop_min_iter, compare='mean')
    else:
        stop_criterion = StopCriteriaImprove(min_improve=cf.stop_threshold, query_len=cf.stop_query_len,
                                             num_min_iter=cf.stop_min_iter)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=cf.tb_dir, comment=f'Experiment: {cf.experiment_name}')

    global_step = 0
    # Epoch iterations
    for epoch in range(cf.n_epochs):
        print(f'Epoch: {epoch + 1}...')
        net.train()

        # Iterations in one epoch
        pbar = tqdm.tqdm(train_loader, total=n_train, desc=f'Epoch {epoch + 1}/{cf.n_epochs}', unit='group')
        for input_imgs in pbar:
            total_loss = 0
            input_imgs = input_imgs['images'].to(device=device)  # (1,phase_num/group,z,x,y)

            res = net(input_imgs)

            simi_loss = simi_loss_func(res['warped_input'], res['template'])
            total_loss += simi_loss

            if cf.smooth_loss_type is not None:
                smooth_loss = smooth_loss_func(res['dvf'], res['template'])
                total_loss += smooth_loss
                smooth_loss_item = smooth_loss.item()
            else:
                smooth_loss_item = 0

            # Enforce the template locating at the center of all input images
            if cf.cyclic_reg > 0:
                cyclic_loss = (torch.mean((torch.sum(res['dvf'], 0)) ** 2)) ** 0.5  # Reduce (sum along) group dimension
                total_loss += cf.cyclic_reg * cyclic_loss
                cyclic_loss_item = cyclic_loss.item()
            else:
                cyclic_loss_item = 0

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_step += 1

            pbar.set_postfix(
                {'simi_loss': simi_loss.item(), 'smooth_loss': smooth_loss_item, 'cyclic_loss': cyclic_loss_item})
            tb_writer.add_scalar('Train/Simi_loss', simi_loss.item(), global_step)
            tb_writer.add_scalar('Train/Smooth_loss', smooth_loss_item, global_step)
            tb_writer.add_scalar('Train/Cyc_loss', cyclic_loss_item, global_step)
            tb_writer.add_scalar('Train/Loss', total_loss.item(), global_step)

        scheduler.step()
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation after one iteration
        if cf.valid_model:
            net.eval()
            tot_simi_loss = 0

            for val_imgs in val_loader:
                val_imgs = val_imgs['images'].to(device=device)  # (1,group,z,x,y)
                with torch.no_grad():
                    res = net(val_imgs)
                    tot_simi_loss += simi_loss_func(res['warped_input'], res['template']).item()
                    if ssim_metrics:
                        ssim_val = ssim_metrics(res['warped_input'], res['template']).item()
                        print(f'Validation SSIM metric: {ssim_val}')
                        tb_writer.add_scalar('Validation/SSIM', ssim_val, epoch)

            val_simi_loss = tot_simi_loss / n_val
            print(f'Validation similarity loss ({cf.simi_loss_type}): {val_simi_loss}')
            tb_writer.add_scalar('Validation/Simi_loss', val_simi_loss, epoch)
            if (epoch + 1) % cf.checkpoint_save_freq_epoch == 0:
                record_template = torch.squeeze(res['template'])[cf.volume_size[0]//2, :, :]
                tb_writer.add_images('Validation/Template_img', record_template, epoch, dataformats='HW')

            if cf.val_landmark_evl:
                deform2template = res['dvf']  # (n, 3, d, h, w)
                deform2template_tar = deform2template[[0], ...]  # (1, 3, d, h, w)
                deform2phase_tar = process_disp.inverse_disp(deform2template_tar)

                composed_dvf = process_disp.compose_disp(deform2template, deform2phase_tar, mode='all')
                error_mean = landmark_reg_error(composed_dvf, cf.val_data_path, cf.valid_spacing)
                tb_writer.add_scalar('Validation/Error_landmark', error_mean, epoch)

        # Using stop criteria
        if cf.earlyStopping_enabled:
            best_update = stop_criterion.add(total_loss.item())
            # Save best model
            if cf.save_best_model and best_update:
                print(f'Update the best model at epoch {epoch + 1}...')
                torch.save(net.state_dict(), os.path.join(cf.weights_save_path, f'best_model.pth'))
            if stop_criterion.stop():
                break

        # Save weights
        if cf.checkpoint_enabled:
            if (epoch + 1) % cf.checkpoint_save_freq_epoch == 0:
                torch.save(net.state_dict(), os.path.join(cf.weights_save_path, f'CP_epoch{epoch + 1}.pth'))
                print(f'Training loss: {total_loss.item()}')
                print(f'Checkpoint {epoch + 1} saved !')

    torch.save(net.state_dict(), os.path.join(cf.weights_save_path, f'final_weights.pth'))
    print('\n > Training finished.')


def groupwise_test(device, cf):
    # Load test data
    test_dataset = GroupDataset(cf.test_data_path, cf.phase_num)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, worker_init_fn=np.random.seed(0))
    n_test = len(test_loader)
    dim = len(cf.volume_size)

    # Network
    net = RegGroup(image_shape=cf.volume_size, group_num=cf.phase_num, cf=cf)
    net.to(device=device)
    print("\n > Loading pre-trained weights: " + cf.load_weights_path)
    net.load_state_dict(torch.load(cf.load_weights_path, map_location=device))

    process_disp = ProcessDisp(disp_shape=cf.volume_size, calc_device=device)
    simi_loss_func = loss_function(dim, cf, device)
    phase = cf.phase_num

    tot_simi_loss = 0
    test_index = 0
    for test_imgs in test_loader:
        test_imgs = test_imgs['images'].to(device=device)

        with torch.no_grad():
            res = net(test_imgs)
            tot_simi_loss += simi_loss_func(res['warped_input'], res['template']).item()

        # Evaluation for DIR-Lab landmark
        if cf.test_landmark_evl:
            print('Evaluate the landmark registration error between phase 5 and phase 0.')
            deform2template = res['dvf']
            deform2template_tar = deform2template[[0], ...]
            deform2phase_tar = process_disp.inverse_disp(deform2template_tar)

            composed_dvf = process_disp.compose_disp(deform2template, deform2phase_tar, mode='all')
            landmark_reg_error(composed_dvf, cf.test_data_path, cf.test_spacing)
            del deform2template, deform2template_tar, deform2phase_tar, composed_dvf

        # Calculate the interphase DVFs (between different phases rather than template)
        if len(cf.target_indexes) > 0:
            deform2template = res['dvf']  # (n, 3, d, h, w)
            deform2template_tar = deform2template[cf.target_indexes, ...]  # (m, 3, d, h, w), m = len(target_indexes)
            deform2phase_tar = process_disp.inverse_disp(deform2template_tar)

            composed_dvf = process_disp.compose_disp(deform2template, deform2phase_tar, mode='all')

            # Save interphase DVFs and image deformation results
            for j in range(len(cf.target_indexes)):
                temp_dvf = composed_dvf[:, j, ...]
                for i in range(phase):
                    pair_dvf = temp_dvf[i, ...]     # (3,d,h,w)
                    if cf.save_dvf2phase:
                        pair_dvf_save = np.moveaxis(pair_dvf.cpu().numpy(), 0, -1)
                        if cf.target_indexes[j] == i:
                            pair_dvf_save *= 0

                        save_as_itk(np.flip(pair_dvf_save, axis=-1), os.path.join(cf.test_save_path, 'Case' +
                                    str(test_index) + '_dvf2phase' + str(cf.target_indexes[j]) + '_' + str(i) + '.mha'),
                                    spacing=cf.test_spacing, isVector=True)
                    if cf.save_deformed_img:
                        mv_img = test_imgs.squeeze()[i, ...]
                        deformed_image = process_disp.spatial_transform(mv_img[np.newaxis, np.newaxis, ...], pair_dvf[np.newaxis, ...])
                        save_as_itk(deformed_image.squeeze().cpu().numpy(),
                                    os.path.join(cf.test_save_path, 'Case' + str(test_index) + '_deform2phase' +
                                                 str(cf.target_indexes[j]) + '_' + str(i) + '.mha'), spacing=cf.test_spacing)

        # Save phase-to-template result
        input_images = test_imgs.squeeze().cpu().numpy()
        warped_input_images = res['warped_input'].squeeze().cpu().numpy()
        template_image = res['template'].squeeze().cpu().numpy()
        dvf_template = res['dvf'].squeeze().cpu().numpy()

        if cf.save_input_itk:
            for p in range(phase):
                save_as_itk(input_images[p, ...], os.path.join(cf.test_save_path, 'Case' + str(test_index) +
                                                               '_input_phase' + str(p) + '.mha'), spacing=cf.test_spacing)

        if cf.save_template_related:
            save_as_itk(template_image, os.path.join(cf.test_save_path, 'Case' + str(test_index) + '_template.mha'),
                        spacing=cf.test_spacing)

            for p in range(phase):
                save_as_itk(warped_input_images[p, ...],
                            os.path.join(cf.test_save_path, 'Case' + str(test_index) + '_deform2template_' + str(p) +
                                         '.mha'), spacing=cf.test_spacing)

                template_dvf_save = np.moveaxis(dvf_template[p, ...], 0, -1)
                save_as_itk(np.flip(template_dvf_save, axis=-1), os.path.join(cf.test_save_path, 'Case' +
                            str(test_index) + '_dvf2template_' + str(p) + '.mha'), spacing=cf.test_spacing, isVector=True)
        test_index += 1

    ave_simi_loss = tot_simi_loss / n_test
    print(f'Testing similarity loss ({cf.simi_loss_type}): {ave_simi_loss}')
    print('\n > Testing finished.')
