CSE 5370 Final Project
Prit Desai UTA ID: 1002170533
To run the Project Run all the cells in main.ipynb till:
torch.cuda.empty_cache()

%run {GENERAL_DIR}/train.py \
--STUDENT_ID {STUDENT_ID} \
--everything_seed {STUDENT_ID} \
--logging_dir {LOGGING_DIR} \
--logging_name {LOGGING_NAME} \
--data_dir {DATA_PATH} \
--patch_size 256 \
--n_patches_per_image 50 \
--whitespace_threshold 0.82 \
--num_dataset_workers 4 \
--split_ratio 0.8 0.1 0.1 \
--normalize_transform \
--num_dataloader_workers 4 \
--batch_size 4 \
--input_height 256 \
--enc_type "resnet18" \
--enc_out_dim 512 \
--kl_coeff 0.001 \
--latent_dim 32 \
--lr 1e-5 \
--accelerator "gpu" \
--strategy "auto" \
--devices 1 \
--max_epochs 20
Try to run this more than once to make sure it runs as it might take a few attemps to run.


I have put the screenshots of the succesfull run below and the results for the succesfull run.


![image](https://github.com/user-attachments/assets/c0e60818-979e-4b62-be32-a4ba9ba49f5d)

![image](https://github.com/user-attachments/assets/ec0c9fc9-3a6e-496f-b182-9ac59af21727)

![image](https://github.com/user-attachments/assets/4d564260-335a-4a1f-9cfd-7d9557069542)

![image](https://github.com/user-attachments/assets/06629179-15c6-4fa0-a48f-4591adc5f39a)


![image](https://github.com/user-attachments/assets/1dc9eef1-4d18-441e-b3bc-a57d3fea686b)

![image](https://github.com/user-attachments/assets/62fe2fa2-44e5-4222-85a1-96938bf85408)

![image](https://github.com/user-attachments/assets/7e48eecd-f4eb-4a4d-ad5d-baf11f8cfe17)

INFO:lightning_fabric.utilities.seed:Seed set to 1002170533
INFO:pytorch_lightning.utilities.rank_zero:You are using the plain ModelCheckpoint callback. Consider using LitModelCheckpoint which with seamless uploading to Model registry.
INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
drive/MyDrive/vae-4/data/006984cf-35fc-4bc3-a0db-7f5b68de4a6e.svs loaded to be patched...
drive/MyDrive/vae-4/data/6bd7f723-3737-491d-ac2f-279eac24f50c.svs loaded to be patched...

drive/MyDrive/vae-4/data/77c41662-2141-474f-9799-97fb74401162.svs loaded to be patched...


Multiprocessing started...
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Multiprocessing ended successfully...
150 patches were created and written successfully in drive/MyDrive/vae-4/data...
There are 120 patches in training set...
There are 15 patches in validation set...
There are 15 patches in test set...


mean of training set used for normalization: [0.76420075 0.53747427 0.6695537 ]
std of training set used for normalization: [0.15980072 0.22202258 0.17146608]


150 patches were loaded successfully from drive/MyDrive/vae-4/data...
There are 120 patches in training set...
There are 15 patches in validation set...
There are 15 patches in test set...
150 patches were loaded successfully from drive/MyDrive/vae-4/data...
There are 120 patches in training set...
There are 15 patches in validation set...
There are 15 patches in test set...
INFO:pytorch_lightning.callbacks.model_summary:
  | Name    | Type          | Params | Mode  | In sizes         | Out sizes       
----------------------------------------------------------------------------------------
0 | encoder | ResNetEncoder | 11.2 M | train | [1, 3, 256, 256] | [1, 512]        
1 | decoder | ResNetDecoder | 6.8 M  | train | [1, 32]          | [1, 3, 256, 256]
2 | fc_mu   | Linear        | 16.4 K | train | [1, 512]         | [1, 32]         
3 | fc_var  | Linear        | 16.4 K | train | [1, 512]         | [1, 32]         
----------------------------------------------------------------------------------------
18.0 M    Trainable params
0         Non-trainable params
18.0 M    Total params
71.969    Total estimated model params size (MB)
147       Modules in train mode
0         Modules in eval mode
/usr/local/lib/python3.11/dist-packages/torch/jit/_trace.py:1304: TracerWarning: Trace had nondeterministic nodes. Did you forget call .eval() on your model? Nodes:
	%eps : Float(1, 32, strides=[32, 1], requires_grad=0, device=cuda:0) = aten::normal(%1306, %1314, %1315) # /usr/local/lib/python3.11/dist-packages/torch/distributions/utils.py:61:0
This may cause errors in trace checking. To disable trace checking, pass check_trace=False to torch.jit.trace()
  _check_trace(
/usr/local/lib/python3.11/dist-packages/torch/jit/_trace.py:1304: TracerWarning: Output nr 1. of the traced function does not match the corresponding output of the Python function. Detailed error:
Tensor-likes are not close!

Mismatched elements: 196549 / 196608 (100.0%)
Greatest absolute difference: 0.08495993912220001 at index (0, 2, 1, 134) (up to 1e-05 allowed)
Greatest relative difference: 453574.8563739005 at index (0, 2, 56, 165) (up to 1e-05 allowed)
  _check_trace(
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py:310: The number of training batches (30) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
Epoch 19: 100%
 30/30 [00:30<00:00,  0.98it/s, v_num=9]
INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=20` reached.
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]


mean of training set used for normalization: [0.76420075 0.53747427 0.6695537 ]
std of training set used for normalization: [0.15980072 0.22202258 0.17146608]


150 patches were loaded successfully from drive/MyDrive/vae-4/data...
There are 120 patches in training set...
There are 15 patches in validation set...
There are 15 patches in test set...
/usr/local/lib/python3.11/dist-packages/torch/jit/_trace.py:1304: TracerWarning: Output nr 1. of the traced function does not match the corresponding output of the Python function. Detailed error:
Tensor-likes are not close!

Mismatched elements: 196600 / 196608 (100.0%)
Greatest absolute difference: 1.027094453573227 at index (0, 0, 188, 15) (up to 1e-05 allowed)
Greatest relative difference: 88208.0396740296 at index (0, 1, 152, 231) (up to 1e-05 allowed)
  _check_trace(
Testing DataLoader 0: 100%
 3/3 [00:02<00:00,  1.28it/s]







