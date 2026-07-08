imgfiles = [ "yorg", "yaprx_block_dct", "yaprx_block_pca", "yaprx_block_rica", "yaprx_block_k-svd", "yaprx_nsolt" ];
subcaptions = 'abcdef';
srcfolder = "../data/";
dstfolder = "../data/";
for idx = 1:numel(imgfiles)
    imgfile = imgfiles(idx);
    copyfile(srcfolder+imgfile+".png",dstfolder+"fig11"+subcaptions(idx)+"_"+imgfile+".png")
end