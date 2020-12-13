from imports import *
# R. Wiyatno and A. Xu,
# “Maximal Jacobian-based Saliency Map Attack,”
# arXiv:1808.07945 [cs, stat], Aug. 2018, Accessed: Dec. 09, 2020.
# [Online]. Available: https://arxiv.org/abs/1808.07945.

def jsma(name,model,x,y,chunk,num_pix):
  x_aug = []
  idx = 0

  while idx < len(x):
    if (idx + chunk < len(x)):
      x_aug.extend(make_jsma(model,x[idx:idx+chunk],y[idx:idx+chunk],num_pix))
    else:
      x_aug.extend(make_jsma(model,x[idx:idx+chunk],y[idx:idx+chunk],num_pix))
    idx+=chunk
    print(idx)

  output = np.array(x_aug)
  PATH = "/content/drive/My Drive/ML Final Project Files/"  # Randy's Path
  with open(PATH + name + '.npy', 'wb') as f:
    np.save(f, output)
  return output

def make_jsma(model,x,y,num_pix):
    im = torch.clone(
      torch.from_numpy(x)).detach().requires_grad_(True).to(device)
    im_orig = torch.clone(im).detach().to("cpu")
    score_L = np.ones_like(im_orig)
    score_R = np.ones_like(im_orig)
    x_L = torch.clone(im).detach().to(device)
    x_R = torch.clone(im).detach().to(device)
    last_prob = get_probs(model,im,y)
    #print(last_prob.shape)
    for k in range(im.shape[-1]):
      #print("k: " + str(k))
      for j in range(im.shape[-2]):
        x_L = torch.clone(im).detach().to(device)
        x_R = torch.clone(im).detach().to(device)
        #print("j: " + str(j))
        x_L[:,j,k] = 0
        x_R[:,j,k] = 1
        left_prob = get_probs(model,x_L,y)
        right_prob = get_probs(model,x_R,y)
        count = 0
        for idx in range(im.shape[0]):
          #print(y[idx])
          #print(last_prob[idx])
          #print(left_prob[idx])
          #print(right_prob[idx])
          #print("idx: " + str(idx))
          if (left_prob[idx,y[idx]] < right_prob[idx,y[idx]]) and (left_prob[idx,y[idx]] < last_prob[idx,y[idx]]):
            score_L[idx,j,k] = left_prob[idx,y[idx]]
            count += 1
            #print("L: "+str(count))
          elif (right_prob[idx,y[idx]] < left_prob[idx,y[idx]] and right_prob[idx,y[idx]] < last_prob[idx,y[idx]]):
            score_R[idx,j,k] = right_prob[idx,y[idx]]
            count += 1
            #print("R: "+str(count))
    #print(score_L)
    #print(score_R)
    for i in range(num_pix):
      #print(i)
      #print(np.min(score_L))
      #print(np.min(score_R))
      for j in range(im.shape[0]):
        min_flat_L = np.argmin(score_L[j])
        min_row_L = int(np.floor(min_flat_L / 28))
        min_col_L = min_flat_L % 28
        min_flat_R = np.argmin(score_R[j])
        min_row_R = int(np.floor(min_flat_R / 28))
        min_col_R = min_flat_R % 28
        if (score_L[j,min_row_L,min_col_L] < score_R[j,min_row_R,min_col_R]):
          #print("L")
          #print(min_row_L)
          #print(min_col_L)
          im[j,min_row_L,min_col_L] = 0
          score_L[j,min_row_L,min_col_L] = 1
        else:
          #print("R")
          #print(min_row_R)
          #print(min_col_R)
          im[j,min_row_R,min_col_R] = 1
          score_R[j,min_row_R,min_col_R] = 1
    return im.detach().cpu().numpy()