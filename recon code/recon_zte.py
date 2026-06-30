import ismrmrd
import gadgetron
import finufft
import numpy as np
import nibabel as nib
import time
import datetime
import re
import os
import subprocess
import cupy as cp
import cufinufft

def recon(connection):
    
    IMAGE_SIZE = 56
    index = 1
    LCH = 64
    LRO = 200
    second_line = 2280
    first_line = 55 * 2280
    maxline = 1
    DwellTime = 0
    DeadTime = 0
    Segment = 55
    SpokesPerSeg = 2280
    grad_factor = 10
    FOV = 208
    Tra = 0
    Sag = 0
    Cor = 0
    points = 30

    position_correction = False
    new_folder = True
    
    second_line_index = 1
    first_line_index = 1
    image_index = 1

    for acq in connection:
        if (index == 1):
            LCH = acq.active_channels
            LRO = acq.number_of_samples
            FOV = acq.idx.kspace_encode_step_2
            Tra = acq.user_float[1]
            Sag = acq.user_float[2]
            Cor = acq.user_float[3]
            phase_correction = acq.idx.repetition #相位校正
            #phase_correction = False
            
            sigma = 0.2 * LRO
            x = np.arange(1,(LRO + 1))
            window = 1.0 / (1 + np.exp((x - sigma) / (0.05 * LRO)))
            
            traj_factor = np.pi * (LRO / IMAGE_SIZE)
            voxel_spacing = [FOV / IMAGE_SIZE] * 3 
            affine = np.diag(voxel_spacing + [1])  # 4x4 仿射矩阵
            affine[0,0] = -affine[0,0]
            affine[1,1] = -affine[1,1]

            second_line = acq.idx.user[1]
            DwellTime = acq.idx.user[2] / int((1e3 * 2))
            DeadTime = acq.idx.user[3]
            grad_factor = acq.idx.user[4]
            Segment = acq.idx.segment
            SpokesPerSeg = acq.idx.kspace_encode_step_1
            
            first_line = Segment * SpokesPerSeg    
            Repeat = 1                  # 是否进行在线重建，0表示不进行，1表示进行
            
            second_data = np.zeros((LCH,second_line,points),dtype=np.complex64)
            xx2 = np.zeros((second_line,points),dtype=np.float32)
            yy2 = np.zeros((second_line,points),dtype=np.float32)
            zz2 = np.zeros((second_line,points),dtype=np.float32)
            
            first_data = np.zeros((LCH,SpokesPerSeg,LRO),dtype=np.complex64)
            xx1 = np.zeros((SpokesPerSeg,LRO),dtype=np.float32)
            yy1 = np.zeros((SpokesPerSeg,LRO),dtype=np.float32)
            zz1 = np.zeros((SpokesPerSeg,LRO),dtype=np.float32)
            w1 = np.zeros((SpokesPerSeg,LRO),dtype=np.float32)
            
            if(Repeat > 0):
                total_first_data = np.zeros((LCH,first_line,LRO),dtype=np.complex64)
                total_xx1 = np.zeros((first_line,LRO),dtype=np.float32)
                total_yy1 = np.zeros((first_line,LRO),dtype=np.float32)
                total_zz1 = np.zeros((first_line,LRO),dtype=np.float32)
                
                first_data_copy = np.zeros((LCH,SpokesPerSeg,LRO),dtype=np.complex64)
                xx1_copy = np.zeros((SpokesPerSeg,LRO),dtype=np.float32)
                yy1_copy = np.zeros((SpokesPerSeg,LRO),dtype=np.float32)
                zz1_copy = np.zeros((SpokesPerSeg,LRO),dtype=np.float32)
                
                total_sigma = 0.8 * LRO
                total_window = 1.0 / (1 + np.exp((x - total_sigma) / (0.05 * LRO)))
            motion_params = np.zeros(((Segment-1),6))
            
        if(index <= second_line):   # For Second Data Set

            start_index = (second_line_index-1) * points
            end_index = start_index + points
            
            second_data[:,second_line_index-1,:] = acq.data[:,:points]
            second_theta = acq.idx.average / 1e4
            second_phi = acq.idx.contrast / 1e4

            x2 = np.sin(second_theta) * np.cos(second_phi)
            y2 = -np.sin(second_theta) * np.sin(second_phi)
            z2 = -np.cos(second_theta)
        
            i = np.arange(points)
            time_factor2 = (DeadTime + i * DwellTime) / (DeadTime + (LRO -1) * DwellTime)
            traj_x = time_factor2 * x2 / grad_factor * traj_factor
            traj_y = time_factor2 * y2 / grad_factor * traj_factor
            traj_z = time_factor2 * z2 / grad_factor * traj_factor
            
            xx2[second_line_index-1,:] = traj_x
            yy2[second_line_index-1,:] = traj_y
            zz2[second_line_index-1,:] = traj_z
            second_line_index = second_line_index + 1
    
        if(index > second_line):  # For First Data Set
            
            start_index = (first_line_index - 1) * LRO
            end_index = start_index + LRO 
            
            first_data[:,first_line_index - 1,:] = acq.data * window
            first_theta = acq.idx.average / 1e4
            first_phi = acq.idx.contrast / 1e4
        
            x1 = np.sin(first_theta) * np.cos(first_phi)
            y1 = -np.sin(first_theta) * np.sin(first_phi)
            z1 = -np.cos(first_theta)

            i = np.arange(LRO)
            time_factor = (DeadTime + i * DwellTime) / (DeadTime + (LRO -1) * DwellTime)

            traj_x = time_factor * x1 * traj_factor
            traj_y = time_factor * y1 * traj_factor
            traj_z = time_factor * z1 * traj_factor

            xx1[first_line_index-1,:] = traj_x 
            yy1[first_line_index-1,:] = traj_y 
            zz1[first_line_index-1,:] = traj_z 
            w1[first_line_index -1,:] = traj_x**2 + traj_y**2 + traj_z**2
        
            if(Repeat > 0) :
                first_data_copy[:,first_line_index - 1,:] = acq.data * total_window
                xx1_copy[first_line_index - 1,:] = traj_x
                yy1_copy[first_line_index - 1,:] = traj_y
                zz1_copy[first_line_index - 1,:] = traj_z
            first_line_index = first_line_index + 1
            
            if(acq.flags == 134217728): # first_data的最后一条线，准备重建 + 配准
                
                # Last SPE + FeedBack  -->  134217736
                statr_time = time.time() 
                first_data_gpu = cp.asarray(first_data)
                w1_gpu = cp.asarray(w1)
                xx1_gpu = cp.asarray(xx1)
                yy1_gpu = cp.asarray(yy1)
                zz1_gpu = cp.asarray(zz1)
                data_gpu = first_data_gpu.reshape(LCH,LRO * SpokesPerSeg) * (w1_gpu.reshape(-1))
                image = cufinufft.nufft3d1(xx1_gpu.reshape(-1),yy1_gpu.reshape(-1),zz1_gpu.reshape(-1),data_gpu,(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE))
                img_gpu = cp.sum(cp.abs(image)**2,axis=0)**(1/2)
                img = cp.asnumpy(img_gpu)
                img_nifitt = nib.Nifti1Image(img, affine)
                moving_path = f"/home/user/result_image/img{image_index}.nii.gz"
                nib.save(img_nifitt,moving_path)
                if image_index >= 2:
                    base_path = f"/home/user/result_image/img1.nii.gz"
                    tempfile = f"/home/user/result_image/temp.1D"
                    cmd = ["3dvolreg","-base",base_path,"-1Dfile",tempfile,moving_path]
                    result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL, text=True,check=True)
                    line = np.loadtxt(tempfile)
                    motion_params[image_index-2,:] = line   #line中是6个运动参数
                    acq.user_float[0] = line[0]
                    acq.user_float[1] = line[1]
                    acq.user_float[2] = line[2]
                    acq.user_float[3] = line[3]
                    acq.user_float[4] = line[4]
                    acq.user_float[5] = line[5]
                end_time = time.time()
                #print("index : {} , time : {}".format(image_index,end_time-statr_time))
                if image_index >= 2:
                    connection.send(acq)
                    
                if(Repeat > 0):
                    begin = (image_index - 1)  * SpokesPerSeg
                    end  = begin + SpokesPerSeg
                    total_first_data[:,begin : end,:] = first_data_copy
                    total_xx1[begin : end,:] = xx1_copy
                    total_yy1[begin : end,:] = yy1_copy
                    total_zz1[begin : end,:] = zz1_copy
                    first_data_copy[:] = 0
                    xx1_copy[:] = 0
                    yy1_copy[:] = 0
                    zz1_copy[:] = 0
                image_index = image_index + 1
                first_line_index = 1
                first_data[:] = 0
                xx1[:] = 0
                yy1[:] = 0
                zz1[:] = 0
                w1[:] = 0
        index = index + 1
        
    print("--------------------------------------------")
    
    if(Repeat > 0):

        print("Start Total Recon : ")
        
        voxel = (FOV / IMAGE_SIZE)
        total_xx1 = total_xx1 / (LRO / IMAGE_SIZE)
        total_yy1 = total_yy1 / (LRO / IMAGE_SIZE)
        total_zz1 = total_zz1 / (LRO / IMAGE_SIZE)

        if phase_correction :
        
            for i in range(1,Segment)  :
                
                # roll : z, pitch : x, yaw : y
                # dS : dz, dL : x, dP : y
               
                dx = motion_params[i-1][4] / voxel
                dy = -motion_params[i-1][5] / voxel
                dz = -motion_params[i-1][3] / voxel
                
                roll = -np.deg2rad(motion_params[i-1][0]) 
                pitch = np.deg2rad(motion_params[i-1][1])
                yaw = -np.deg2rad(motion_params[i-1][2])
                
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(pitch), -np.sin(pitch)],
                               [0, np.sin(pitch), np.cos(pitch)]])
                Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                               [0, 1, 0],
                               [-np.sin(yaw), 0, np.cos(yaw)]])
                Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                               [np.sin(roll), np.cos(roll), 0],
                               [0, 0, 1]])

                R = Ry @ Rx @ Rz
                
                for j in range(0,SpokesPerSeg) :
                    pindex_start = i * SpokesPerSeg + j 
        
                    x = (total_xx1[pindex_start,:])
                    y = (total_yy1[pindex_start,:])
                    z = (total_zz1[pindex_start,:])
                    
                    k = np.stack([x,y,z],axis=1)
                    k_rot = k @ R
                    
                    x,y,z = k_rot.T
                    
                    H = np.exp(-1j*(x * dx + y * dy + z * dz))
                    datae = total_first_data[:,pindex_start,:] * H
                    total_first_data[:,pindex_start,:] = datae
                    
                    total_xx1[pindex_start,:] = x
                    total_yy1[pindex_start,:] = y
                    total_zz1[pindex_start,:] = z    
    
                # for j in range(0,SpokesPerSeg) :
                    
                #     pindex_start = i * SpokesPerSeg + j 
                #     x = (total_xx1[pindex_start,:])
                #     y = (total_yy1[pindex_start,:])
                #     z = (total_zz1[pindex_start,:])
                #     H = np.exp(-1j*(x * dx + y * dy + z * dz))
                #     datae = total_first_data[:,pindex_start,:] * H
                #     total_first_data[:,pindex_start,:] = datae
                
        if position_correction :
            dz = Tra 
            dx = Sag
            dy = Cor
            for i in range(0,second_line):
                x = xx2[i,:]
                y = yy2[i,:]
                z = zz2[i,:]
                H = np.exp(-1j*(x * dx + y * dy + z * dz))
                second_data[:,i,:] = second_data[:,i,:] * H
        
            for i in range(0,first_line):
                x = total_xx1[i,:]
                y = total_yy1[i,:]
                z = total_zz1[i,:]
                H = np.exp(-1j*(x * dx + y * dy + z * dz))
                total_first_data[:,i,:] = total_first_data[:,i,:] * H
                   
        xx2 = xx2 / (LRO / IMAGE_SIZE)
        yy2 = yy2 / (LRO / IMAGE_SIZE)
        zz2 = zz2 / (LRO / IMAGE_SIZE)
        
        xx2 = xx2.reshape(-1)
        yy2 = yy2.reshape(-1)
        zz2 = zz2.reshape(-1)
        w2 = (xx2**2 + yy2**2 + zz2**2) * (first_line / second_line / grad_factor)
        total_xx1 = total_xx1.reshape(-1)
        total_yy1 = total_yy1.reshape(-1)
        total_zz1 = total_zz1.reshape(-1)
        total_w1 = (total_xx1**2 + total_yy1**2 + total_zz1**2)
        
        second_data = second_data.reshape(LCH,points * second_line)
        total_first_data = total_first_data.reshape(LCH,LRO * first_line)
            
        traj_xx = np.concatenate((xx2,total_xx1),axis=0)
        trai_yy = np.concatenate((yy2,total_yy1),axis=0)
        traj_zz = np.concatenate((zz2,total_zz1),axis=0)
        traj_w = np.concatenate((w2,total_w1),axis=0)
        zte_kdata = np.concatenate((second_data,total_first_data),axis=1) * traj_w

        nufft_type = 1
        plan = finufft.Plan(nufft_type,(LRO,LRO,LRO),n_trans=LCH,eps=1e-6,dtype='complex64')
        plan.setpts(traj_xx,trai_yy,traj_zz)
        zte_image_complex = plan.execute(zte_kdata)
        zte_image = np.sum(np.abs(zte_image_complex)**2,axis=0)**(1/2)

        voxel_spacing = [FOV / LRO] *  3    # FOV 与 LRO 保持一致 
        affine2 = np.diag(voxel_spacing + [1])  # 4x4 仿射矩阵
        affine2[0,0] = -affine2[0,0]
        affine2[1,1] = -affine2[1,1]
        img_nifitt_total = nib.Nifti1Image(zte_image, affine2)
        
        if new_folder:
            today = datetime.datetime.today()
            date_folder = today.strftime("%Y%m%d")
            base_path = "/home/user/result_image"
            date_folder_path = os.path.join(base_path,date_folder)
            if not os.path.exists(date_folder_path):
                os.makedirs(date_folder_path)
            # 4. 扫描日期文件夹中已有的test文件夹，找出最大编号
            existing_folders = []
            if os.path.exists(date_folder_path):
                for item in os.listdir(date_folder_path):
                    item_path = os.path.join(date_folder_path, item)
                    if os.path.isdir(item_path) and item.startswith("test"):
                        existing_folders.append(item)
            
            # 5. 提取已有的test文件夹编号
            max_number = 0
            pattern = re.compile(r'test(\d+)$')
            
            for folder in existing_folders:
                match = pattern.match(folder)
                if match:
                    number = int(match.group(1))
                    if number > max_number:
                        max_number = number
            # 6. 确定下一个文件夹编号
            next_number = max_number + 1
            new_folder_name = f"test{next_number}"
            new_folder_path = os.path.join(date_folder_path, new_folder_name)
            
            # 7. 创建新的test文件夹
            os.makedirs(new_folder_path)
            
            nib.save(img_nifitt_total,os.path.join(new_folder_path,"total.nii.gz"))
            np.save(os.path.join(new_folder_path,"motion_params.npy"),motion_params)
        
        else:
            nib.save(img_nifitt_total,"/home/user/result_image/total.nii.gz")
            np.save("/home/user/result_image/motion_params.npy",motion_params)
        print("End Total Recon!")  