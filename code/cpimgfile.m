imgfiles = [ "yorg", "yaprx_block_dct", "yaprx_block_pca", "yaprx_block_rica", "yaprx_block_k-svd", "yaprx_nsolt" ];
subcaptions = [ 'a', 'b', 'c', 'd', 'e', 'f'];
srcfolder = "../data/";
dstfolder = "../data/";
idx = 1;
for imgfile = imgfiles
    copyfile(srcfolder+imgfile+".png",dstfolder+"fig11"+subcaptions(idx) +"_"+imgfile+".png")
    idx = idx+1;
end