from imports import *
def rdm_down(x):
  index_temp = int(np.floor((x[0].shape[0])/2))
  x_down = np.zeros((1,index_temp,index_temp))
  odd_set = 2*np.arange(index_temp) + 1
  if (odd_set[-1]>x[0].shape[0]-1):
    odd_set = odd_set[:-1]
  even_set = 2*np.arange(int(np.floor((x[0].shape[0])/2)))
  for im in x:
    if (rn.uniform(0,1)>0.5):
      temp = im[odd_set,:]
      temp = temp[:,odd_set]
    else:
      temp = im[even_set,:]
      temp = temp[:,even_set]
    temp = np.expand_dims(temp, axis=0)
    x_down = np.append(x_down,temp,axis=0)
  return x_down[1:]


def rdm_shift(x, x2):
    x_shift = np.zeros((1, x[0].shape[0], x[0].shape[1]))
    x_orig_shift = np.zeros((1, x2[0].shape[0], x2[0].shape[1]))
    for im, orig in zip(x, x2):
        rand1 = rn.randint(1, 3)
        rand2 = rn.randint(1, 3)
        temp = np.roll(im, rand1, axis=0)
        temp = np.roll(temp, rand2, axis=1)
        temp = np.expand_dims(temp, axis=0)
        x_shift = np.append(x_shift, temp, axis=0)

        temp = np.roll(orig, 2 * rand1, axis=0)
        temp = np.roll(temp, 2 * rand2, axis=1)
        temp = np.expand_dims(temp, axis=0)
        x_orig_shift = np.append(x_orig_shift, temp, axis=0)
    return x_shift[1:], x_orig_shift[1:]

def sr_upsample(x):
  up_samp = transforms.Resize(size=(28,28),interpolation=2)
  super_x = np.zeros((1,28,28))
  temp = torch.from_numpy(x)
  for im in temp:
    im = torch.unsqueeze(im,0)
    temp_im = up_samp(im)
    temp_im = temp_im.numpy()
    super_x = np.append(super_x,temp_im,axis=0)
  return super_x[1:]

def rain_sr_batch(model,x,batchsize):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ims = torch.from_numpy(x).float()
  model.to(device)
  model.eval()
  with torch.no_grad():
    if (len(ims.shape)<4):
      ims = torch.unsqueeze(ims,1)
    ret = torch.zeros_like(ims)
    idx = 0
    while idx<ims.shape[0] :
      im_batch = ims[idx:idx+batchsize,:,:,:].to(device)
      ret[idx:idx+batchsize,:,:,:] = model(im_batch).to("cpu")
      if (idx+batchsize>=ims.shape[0]):
        im_batch = ims[idx:ims.shape[0]-1,:,:,:].to(device)
        ret[idx:ims.shape[0]-1,:,:,:] = model(im_batch).to("cpu")
      idx = idx+batchsize
  return np.squeeze(ret.detach().numpy())

def rain_pre_process(x):
  x_rain = rdm_down(x)
  x_rain = rdm_shift(x_rain,x_rain)[0]
  x_rain = sr_upsample(x_rain)
  x_rain = rain_sr_batch(rain_sr_net,x_rain,50)
  return x_rain