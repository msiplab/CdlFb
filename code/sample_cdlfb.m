%[text] # `フィルタバンク理論に基づく畳込み辞書学習`
%[text] # `～ 構造制約を利用した畳込みネットワーク構築 ～`
%[text] ## `村松正吾（新潟大学工学部工学科電子情報通信プログラム）`
%[text] 
%[text] `電子情報通学会　基礎・境界ソサイエティ　Fundamentals Review` 
%[text] [`https://www.jstage.jst.go.jp/article/essfr/17/2/17_116/_article/-char/ja`](https://www.jstage.jst.go.jp/article/essfr/17/2/17_116/_article/-char/ja)
%[text] 
%[text] `動作環境：MATLAB R2025a以降（テキスト形式Live Script、R2026aで動作確認）`
%%
%[text] ### 準備
clear 
close all

nsoltDic = ""; % "nsoltdictionary_20230621232700260" 

isCodegen = false; % コード生成 %[control:checkbox:6a42]{"position":[13,18]}
setup(isCodegen) %[output:5beb9c5c]
%%
%[text] ### パラメータ設定
%[text] - ブロックサイズ 
%[text] - 冗長度
%[text] - スパース度 \
% Block size
szBlk = [ 8 8 ];

% Redundancy ratio for RICA/K-SVD
redundancyRatio = 7/3; 

% Sparsity ratio 
sparsityRatio = 3/64;
%%
%[text] ## 画像の読込
%[text] - $\\mathbf{y}\\in\\mathbb{R}^{N}$ \
% 原画像の準備
file_yorg = "../data/yorg.png";
if ~exist(file_yorg,'file')
    unzip('http://www.ess.ic.kanagawa-it.ac.jp/std_img/monoimage2/Mono-Image2.zip','../results')
    yfull = imread('../results/Mono-Image2/512X512/barbara512.bmp');
    ycrop = yfull(1:192,end-255:end);
    imwrite(ycrop,file_yorg)
end

% 原画像の読み込み
yorg = im2double(imread(file_yorg));
szOrg = size(yorg);
%[text] 画像表示
figure
imshow(yorg);
title('Original image y')
%[text] 零平均化
%ymean = mean(y,"all");
%y = yorg - ymean;
meansubtract = @(x) x-mean(x,"all");
y = meansubtract(yorg);
%%
%[text] ## 離散コサイン変換（DCT）
%[text]  $\[\\mathbf{C}\_M\]\_{k,n}=\\sqrt{\\frac{2}{M}} \\alpha\_k\\cos\\frac{k(n+1∕2)\\pi}{M},\\ k,n=0,1,\\cdots,M-1$
%[text]  $\\alpha\_k=\\left\\{\\begin{array}{ll} \\frac{1}{\\sqrt{2}} & k=0 \\\\1 & k=1,2,\\cdots,M-1\\end{array}\\right.$
%[text] #### 基底画像
%[text]  $\\mathbf{B}\_{k,\\ell}=\\mathbf{C}\_M^{-1}\\mathbf{E}\_{k,\\ell}\\mathbf{C}\_M^{-T},\\ k,\\ell=0,1,\\cdots,M-1$
%[text]  $\\mathbf{E}\_{k,\\ell}= \\mathbf{e}\_k\\mathbf{e}\_\\ell^T$
basisImagesDct = zeros(szBlk(1),szBlk(2),prod(szBlk));
iBasis = 1;
for iRow=1:szBlk(1)
    for iCol=1:szBlk(2)
        E = zeros(szBlk);
        E(iRow,iCol) = 1;
        basisImagesDct(:,:,iBasis) = idct2(E,szBlk(1),szBlk(2));
        iBasis = iBasis + 1;
    end
end
%[text] #### 基底画像の表示
figure
montage(imresize(basisImagesDct,8,'nearest')+.5,'BorderSize',[2 2])
title('Basis images of DCT')
%[text] #### ブロックDCTによる合成処理とその随伴処理の定義
syn_blkdct = @(x) blockproc(x,szBlk,@(block_struct) idct2(block_struct.data));
adj_blkdct = @(y) blockproc(y,szBlk,@(block_struct) dct2(block_struct.data));
%[text] 随伴関係の確認
x = adj_blkdct(y);
v = randn(size(x));
u = syn_blkdct(v);
assert(abs(dot(y(:),u(:))-dot(x(:),v(:)))<1e-9)
%%
%[text] ## 主成分分析（PCA）
%[text] 
%[text] #### 問題設定:
%[text] `直交性と次元削減`
%[text]  $\\mathbf{\\Phi}^\\textsf{T} \\mathbf{\\Phi}=\\mathbf{I}\_{M}, \\forall b, \\forall p,\\Vert{\\mathbf{x}\_b}\\Vert\_0\\le p\<M$
%[text] `を制約条件とした最小自乗問題`
%[text]  $\\{\\hat{\\mathbf{\\Phi}},\\{ \\hat{\\mathbf{x}}\_b \\}\_b\\}=\\arg\\min\_{\\{\\mathbf{\\Phi},\\{\\mathbf{x}\_b\\}\_b\\}}\\frac{1}{2S}\\sum\_{b=1}^{S}\\|\\mathbf{y}\_b-\\mathbf{\\Phi}{\\mathbf{x}}\_b\\|\_2^2$
%[text] `を解く．上式は等価的に`
%[text]  $\\hat{\\mathbf{\\Phi}}=\\arg \\max \_{\\mathbf{\\Phi}} \\mathrm{tr}\\left(\\mathbf{\\Phi}\_{:, 0:p-1}^\\textsf{T}\\hat{\\mathbf{\\Sigma}}\_{y} \\mathbf{\\Phi}\_{:,0:p-1}\\right)\\ \\mathrm{ s.t. }\\ \\mathbf{\\Phi}^\\textsf{T} \\mathbf{\\Phi}=\\mathbf{I}\_{M}$
%[text] `と表現できる．`ただし， $\\widehat{\\mathbf{\\Sigma}}\_{y}$は 観測ベクトル $\\{\\mathbf{y}\_b\\}\_b$ （零平均を仮定）の標本分散共分散行列である．
%[text] 
%[text] #### 解:
%[text] 固有値分解
%[text]  $\\widehat{\\mathbf{\\Phi}}^\\textsf{T}\\widehat{\\mathbf{\\Sigma}\_y}\\widehat{\\mathbf{\\Phi}}=\\mathbf{\\Lambda}\n$
%[text] ただし， $\\mathbf{\\Lambda}=\\mathrm{diag}(\\lambda\_1,\\lambda\_2,\\cdots,\\lambda\_M)$． $\\lambda\_1\\geq\\lambda\_2\\geq\\cdots\\lambda\_M$ は $\\widehat{\\mathbf{\\Sigma}}\_{y}$の固有値．
%[text] 
%[text] #### 画像 $\\mathbf{y}$からのデータ行列 $\\mathbf{Y}$ の生成 
%[text] 標本平均ブロックを引く代わりに，予め零平均化したデータで学習
nPatches = 20*prod(szOrg./szBlk); % PCA/RICA/K-SVD 学習用のパッチをランダム抽出
npos = randsample(prod(szOrg-szBlk),nPatches);
ybs = zeros(szBlk(1),szBlk(2),nPatches,'like',y);
szSrchy = szOrg(1)-szBlk(1);
for iPatch = 1:nPatches
    ny_ = mod(npos(iPatch)-1,szSrchy)+1;
    nx_ = floor((npos(iPatch)-1)/szSrchy)+1;
    ybs(:,:,iPatch) = y(ny_:ny_+szBlk(1)-1,nx_:nx_+szBlk(2)-1);
end
figure
montage(ybs+0.5,'Size',[8 8]);
drawnow

Y = reshape(ybs,prod(szBlk),[]);

%[text] 標本分散共分散行列 $\\widehat{\\mathbf{\\Sigma}}\_{y}$の計算 
SigmaY = cov(Y.');
%[text] 標本分散共分散行列 $\\widehat{\\mathbf{\\Sigma}}\_{y}$の固有値分解 
[Phi_pca,Lambda] = eig(SigmaY);
%[text] 固有値 $\\lambda$ の大きさの降順に列ベクトルをソート (Sorting column vectors in the descending order of the eigenvalues $\\lambda$)
[~,idx] = sort(diag(Lambda),'descend');
Phi_pca = Phi_pca(:,idx);
%[text] 固有ベクトルを基底画像に変換
nBases = prod(szBlk);
basisImagesPca = zeros(szBlk(1),szBlk(2),nBases);
for iBasis = 1:nBases
    basisImagesPca(:,:,iBasis) = reshape(Phi_pca(:,iBasis),szBlk(1),szBlk(2));
end
%[text] #### 基底画像の表示（辞書）
figure
montage(imresize(basisImagesPca,8,'nearest')+.5,'BorderSize',[2 2])
title('Basis images of PCA(KLT)')
%[text] #### ブロックPCAによる合成処理とその随伴処理の定義
syn_blkpca = @(x) col2im(Phi_pca*x,szBlk,szOrg,"distinct");
adj_blkpca = @(y) Phi_pca.'*im2col(y,szBlk,"distinct");
%[text] 随伴関係の確認
x = adj_blkpca(y);
v = randn(size(x));
u = syn_blkpca(v);
assert(abs(dot(y(:),u(:))-dot(x(:),v(:)))<1e-9)
%%
%[text] ## 再構成独立成分分析（RICA）
%[text] 
%[text] #### 問題設定:
%[text]  $\\widehat{\\mathbf{\\Phi}}=\\arg \\min \_{\\mathbf{\\Phi}} \\frac{1}{2S}\\sum\_{b=1}^S\\|\\mathbf{y}\_b-\\mathbf{\\Phi}\\mathbf{\\Phi}^\\textsf{T}\\mathbf{y}\_b\\|\_2^2+\\frac{\\alpha}{S}\\sum\_{b=1}^{S}\\rho(\\mathbf{\\Phi}^\\textsf{T}\\mathbf{y}\_b)\n$
%[text]  $=\\arg \\min \_{\\mathbf{\\Phi}} \\frac{(2\\alpha)^{-1}}{S}\\sum\_{b=1}^S\\|\\mathbf{y}\_b-\\mathbf{\\Phi}\\mathbf{\\Phi}^\\textsf{T}\\mathbf{y}\_b\\|\_2^2+\\frac{1}{S}\\sum\_{b=1}^{S}\\rho(\\mathbf{\\Phi}^\\textsf{T}\\mathbf{y}\_b)$
%[text] ただし，  $\\{\\mathbf{y}\_n\\}\_n\\subset\\mathbb{R}^{M}$,  $\\mathbf{\\Phi}=(\\mathbf{\\phi}\_1,\\mathbf{\\phi}\_2,\\cdots,\\mathbf{\\phi}\_P)\\in\\mathbb{R}^{M\\times P}$, $M\\geq P$ である．
%[text] 
%[text] #### 参考文献:
%[text] Le, Quoc V., Alexandre Karpenko, Jiquan Ngiam, and Andrew Y. Ng. “ICA with Reconstruction Cost for Efficient Overcomplete Feature Learning.” Advances in Neural Information Processing Systems. Vol. 24, 2011, pp. 1017–1025. https://papers.nips.cc/paper/4467-ica-with-reconstruction-cost-for-efficient-overcomplete-feature-learning.pdf. 
%[text] 
%[text] パラメータ設定
%[text] - 繰返し回数 (Number of iterations)
%[text] - 正則化パラメータ (Regularization parameter) \
% Number of iterations
nItersRica = 1e5; 
% Regularization parameter
alpha = 2e-3;
%[text] コントラスト関数の例
%[text]  $\\rho(\\mathbf{\\Phi}^\\textsf{T}\\mathbf{y})\\colon = \\frac{1}{2}\\sum\_{p=1}^{P}\\log\\circ\\cosh(2\\mathbf{\\phi}\_p^\\textsf{T}\\mathbf{y})$
figure
fplot(@(x) abs(x),[-5 5],':','LineWidth',2,'DisplayName','|\cdot|')
hold on
fplot(@(x) log(cosh(2*x))/2,[-5 5],'-','LineWidth',2,'DisplayName','1/2log(cosh(2\cdot))')
xlabel('x')
legend
grid on
axis equal
hold off
%[text] 要素画像の数 
nDims = prod(szBlk);
nAtoms = ceil(redundancyRatio*nDims);
%[text] 辞書 $\\mathbf{\\Phi}$の初期化
%[text] - 二次元離散コサイン変換
%[text] - ランダム \
Phi_rica = randn(nDims,nAtoms);
Phi_rica = Phi_rica/norm(Phi_rica,'fro');
for iAtom = 1:nDims
    delta = zeros(szBlk);
    delta(iAtom) = 1;
    Phi_rica(:,iAtom) = reshape(idct2(delta),nDims,1);
end
%[text] 要素ベクトルを要素画像に変換 
atomicImagesRica = zeros(szBlk(1),szBlk(2),nAtoms);
for iAtom = 1:nAtoms
    atomicImagesRica(:,:,iAtom) = reshape(Phi_rica(:,iAtom),szBlk(1),szBlk(2));
end
figure
montage(imresize(atomicImagesRica,8,'nearest')+.5,'BorderSize',[2 2],'Size',[ceil(nAtoms/8) 8])
title('Atomic images of initial dictionary (DCT & random)')
%[text] #### 再構成 ICA オブジェクトの作成
%[text] PCAに合わせて予め零平均化したデータで学習
model = rica(Y.',nAtoms,...
    'IterationLimit',nItersRica,...
    'ContrastFcn','logcosh',...
    'InitialTransformWeight',Phi_rica,...
    'Lambda',1/(2*alpha));
%[text] コスト評価のグラフ 
info = model.FitInfo;
figure
plot(info.Iteration,info.Objective)
xlabel('Number of iteration')
ylabel('Cost')
grid on
%[text] 要素ベクトルを要素画像に変換
Phi_rica = model.TransformWeights;
atomicImagesRica = zeros(szBlk(1),szBlk(2),nAtoms);
for iAtom = 1:nAtoms
    atomicImagesRica(:,:,iAtom) = reshape(Phi_rica(:,iAtom),szBlk(1),szBlk(2));
end
%[text] #### 要素画像の表示（辞書）
figure
montage(imresize(atomicImagesRica,8,'nearest')+.5,'BorderSize',[2 2],'Size',[ceil(nAtoms/8) 8])
title('Atomic images of RICA')
%[text] #### ブロックRICAによる合成処理とその随伴処理の定義
syn_blkrica = @(x) col2im(Phi_rica*x,szBlk,szOrg,"distinct");
adj_blkrica = @(y) Phi_rica.'*im2col(y,szBlk,"distinct");
%[text] 随伴関係の確認
x = adj_blkrica(y);
v = randn(size(x));
u = syn_blkrica(v);
assert(abs(dot(y(:),u(:))-dot(x(:),v(:)))<1e-9)
%%
%[text] ## K-特異値分解
%[text] パラメータ設定
%[text] - 繰返し回数 (Number of iterations) \
% Number of iterations
nItersKsvd = 5e3;
%[text] #### 問題設定 (Problem setting):
%[text]  $\\{\\hat{\\mathbf{\\Phi}},\\{ \\hat{\\mathbf{x}}\_b \\}\\}=\\arg\\min\_{\\{\\mathbf{\\Phi},\\{\\mathbf{x}\_b\\}\\}}\\frac{1}{2S}\\sum\_{b=1}^{S}\\|\\mathbf{y}\_b-\\mathbf{\\Phi}\\hat{\\mathbf{x}}\_b\\|\_2^2,\\ \\quad\\mathrm{s.t.}\\ \\forall b, \\|\\mathbf{x}\_b\\|\_0\\leq K&dollar&;$
%[text] #### アルゴリズム :
%[text] スパース近似ステップと辞書更新ステップを繰返す．
%[text] - スパース近似ステップ  \
%[text]  $\\hat{\\mathbf{x}}\_b=\\arg\\min\_{\\mathbf{x}} \\frac{1}{2}\\|\\mathbf{y}\_b-\\hat{\\mathbf{\\Phi}}\\mathbf{x}\\|\_2^2\\ \\quad \\mathrm{s.t.}\\ \\|\\mathbf{x}\\|\_0\\leq K$
%[text] - 辞書更新ステップ  \
%[text]  $\\hat{\\mathbf{\\Phi}}=\\arg\\min\_{\\mathbf{\\Phi}}\\frac{1}{2S}\\sum\_{b=1}^{S}\\|\\mathbf{y}\_b-\\mathbf{\\Phi}\\hat{\\mathbf{x}}\_b\\|\_2^2=\\arg\\min\_{\\mathbf{\\Phi}}\\frac{1}{2S}\\left\\|\\left(\\mathbf{Y}-\\sum\_{p\\neq k}\\mathbf{\\phi}\_p\\hat{\\mathbf{X}}\_{p,\\colon}\\right)-\\mathbf{\\phi}\_k\\hat{\\mathbf{X}}\_{k,\\colon}\\right\\|\_F^2$
%[text] 
%[text] 係数の数 
nCoefsKsvd = max(floor(sparsityRatio*nDims),1);
%[text] 辞書 $\\mathbf{\\Phi}$の初期化 
%[text] - 二変量離散コサイン変換
%[text] - ランダム  \
Phi_ksvd = randn(nDims,nAtoms);
Phi_ksvd = Phi_ksvd/norm(Phi_ksvd,'fro');
for iAtom = 1:nDims
    delta = zeros(szBlk);
    delta(iAtom) = 1;
    Phi_ksvd(:,iAtom) = reshape(idct2(delta),nDims,1);
end
%[text] 要素ベクトルを要素画像に変換
atomicImagesKsvd = zeros(szBlk(1),szBlk(2),nAtoms);
for iAtom = 1:nAtoms
    atomicImagesKsvd(:,:,iAtom) = reshape(Phi_ksvd(:,iAtom),szBlk(1),szBlk(2));
end
figure
montage(imresize(atomicImagesKsvd,8,'nearest')+.5,'BorderSize',[2 2],'Size',[ceil(nAtoms/8) 8])
title('Atomic images of initial dictionary (DCT & random)')
%[text] #### スパース近似ステップと辞書更新ステップの繰り返し
%[text] - スパース近似： 直交マッチング追跡 (OMP)
%[text] - 辞書更新： 特異値分解(SVD)と1-ランク近似  \
%[text] 辞書更新の内容
%[text] 1. $k\\leftarrow 1$
%[text] 2. 誤差行列 $\\mathbf{E}\_k$ を定義：$\\mathbf{E}\_k\\colon = \\mathbf{Y}-\\sum\_{p\\neq k}\\mathbf{\\phi}\_p\\hat{\\mathbf{X}}\_{p,\\colon}$
%[text] 3. データ行 $\\hat{\\mathbf{X}}\_{k,\\colon}$の非零値を抽出する行列 $\\mathbf{\\Omega}\_k$を定義： $\\hat{\\mathbf{X}}\_{k,\\colon}^R=\\hat{\\mathbf{X}}\_{k,\\colon}\\mathbf{\\Omega}\_k \\Leftrightarrow \\hat{\\mathbf{X}}\_{k,\\colon}^R\\mathbf{\\Omega}\_k^T=\\hat{\\mathbf{X}}\_{k,\\colon}$
%[text] 4. 誤差行列 $\\mathbf{E}\_k$ を行列 $\\mathbf{\\Omega}\_k$で縮退： $\\mathbf{E}\_k^R \\colon=\\mathbf{E}\_k\\mathbf{\\Omega}\_k$
%[text] 5. 縮退した誤差行列$\\mathbf{E}\_k^R$を特異値分解：$\\mathbf{E}\_k^R =\\mathbf{U}\\mathbf{S}\\mathbf{V}^T\n=\\left(\\mathbf{u}\_1,\\mathbf{u}\_2,\\cdots,\\mathbf{u}\_r\\right)\\mathrm{diag}(\\sigma\_1,\\sigma\_2,\\cdots,\\sigma\_r)\\left(\\mathbf{v}\_1,\\mathbf{v}\_2,\\cdots,\\mathbf{v}\_r\\right)^T$
%[text] 6. 要素ベクトル $\\mathbf{\\phi}\_k$ を更新： $\\mathbf{k}\\leftarrow \\mathbf{u}\_1$
%[text] 7. データ行$\\hat{\\mathbf{X}}\_{k,\\colon}$を更新： $\\hat{\\mathbf{X}}\_{k,\\colon}\\leftarrow \\sigma\_1\\mathbf{v}\_{1}^T$
%[text] 8. $k\\leftarrow k+1$
%[text] 9. $k\\leq N$ ならば 2. へ $k\>N$ ならば終了 \
%[text] ただし， $\\sigma\_1$ を最大特異値とする．
%[text] #### 交互ステップの繰返し計算
%[text] PCAに合わせて予め零平均化したデータで学習
cost = zeros(1,nItersKsvd);
nSamples = size(Y,2);
for iIter = 1:nItersKsvd
    X = zeros(nAtoms,nSamples);
    % Sparse approximation
    for iSample = 1:nSamples
        y_ = Y(:,iSample);
        x = omp(y_,Phi_ksvd,nCoefsKsvd);
        X(:,iSample) = x;
    end
    % Dictionary update
    % R is kept as the running residual Y-Phi_ksvd*X so that each atom's
    % error matrix can be recovered by adding back its own contribution,
    % avoiding an O(nAtoms) setdiff and a full (nAtoms-1)-column matrix
    % product on every iteration.
    R = Y - Phi_ksvd*X;
    for iAtom = 1:nAtoms
        xk = X(iAtom,:);
        suppk = find(xk);
        %
        if ~isempty(suppk)
            Ekred = R(:,suppk) + Phi_ksvd(:,iAtom)*xk(suppk);
            [U,S,V] = svd(Ekred,'econ');
            ak = U(:,1);
            xkred = S(1,1)*V(:,1)';
            %
            Phi_ksvd(:,iAtom) = ak;
            X(iAtom,suppk) = xkred;
            R(:,suppk) = Ekred - ak*xkred;
        end
    end
    cost(iIter) = (norm(Y-Phi_ksvd*X,'fro')^2)/(2*nSamples);
end
%[text] コスト評価のグラフ 
figure
plot(cost)
xlabel('Number of iteration')
ylabel('Cost')
grid on
%[text] 要素ベクトルを要素画像に変換 
atomicImagesKsvd = zeros(szBlk(1),szBlk(2),nAtoms);
for iAtom = 1:nAtoms
    atomicImagesKsvd(:,:,iAtom) = reshape(Phi_ksvd(:,iAtom),szBlk(1),szBlk(2));
end
figure
montage(imresize(atomicImagesKsvd,8,'nearest')+.5,'BorderSize',[2 2],'Size',[ceil(nAtoms/8) 8])
title('Atomic images of K-SVD')
%[text] #### ブロックK-特異値分解による合成処理とその随伴処理の定義
syn_blkksvd = @(x) col2im(Phi_ksvd*x,szBlk,szOrg,"distinct");
adj_blkksvd = @(y) Phi_ksvd.'*im2col(y,szBlk,"distinct");
%[text] 随伴関係の確認
x = adj_blkksvd(y);
v = randn(size(x));
u = syn_blkksvd(v);
assert(abs(dot(y(:),u(:))-dot(x(:),v(:)))<1e-9)
%%
%[text] ## 2変量ラティス構造冗長フィルタバンク
%[text] 例として，（偶対称チャネルと奇対称チャネルが等しい）偶数チャネル、偶数のポリフェーズ次数をもつタイプI非分離冗長重複変換(NSOLT)
%[text]  $\\mathbf{E}(z\_\\mathrm{v},z\_\\mathbf{h})\n=\n\\left(\\prod\_{n\_\\mathrm{h}=1}^{\\nu\_\\mathrm{h}/2}\n{\\mathbf{V}\_{2n\_\\mathrm{h}}^{\\{\\mathrm{h}\\}}}\\bar{\\mathbf{Q}}(z\_\\mathrm{h}){\\mathbf{V}\_{2k\_\\mathrm{h}-1}^{\\{\\mathrm{h}\\}}}{\\mathbf{Q}}(z\_\\mathrm{h})\\right)\n%\n\\left(\\prod\_{n\_{\\mathrm{v}}=1}^{\\nu\_\\mathrm{v}/2}{\\mathbf{V}\_{2n\_\\mathrm{v}}^{\\{\\mathrm{v}\\}}}\\bar{\\mathbf{Q}}(z\_\\mathrm{v}){\\mathbf{V}\_{2n\_\\mathrm{v}-1}^{\\{\\mathrm{v}\\}}}{\\mathbf{Q}}(z\_\\mathrm{v})\\right)\n%\n\\mathbf{V}\_0\\mathbf{E}\_0,$
%[text]  $\\mathbf{R}(z\_\\mathrm{v},z\_\\mathbf{h})\n=\\mathbf{E}^\\textsf{T}(z\_\\mathrm{v}^{-1},z\_\\mathrm{h}^{-1}),$
%[text] を採用する．ただし，
%[text] - $\\mathbf{E}(z\_\\mathrm{v},z\_\\mathrm{h})$:  分析フィルタバンクのType-I ポリフェーズ行列
%[text] - $\\mathbf{R}(z\_\\mathrm{v},z\_\\mathrm{h})$: 合成フィルタバンクのType-II ポリフェーズ行列
%[text] - $z\_d\\in\\mathbb{C}, d\\in\\{\\mathrm{v},\\mathrm{h}\\}$: Z-変換の変数
%[text] - $\\nu\_d\\in \\mathbb{N}, d\\in\\{\\mathrm{v},\\mathrm{h}\\}$:方向 $d$ のポリフェーズ次数(重複ブロック数)
%[text] - $\\mathbf{V}\_0=\\left(\\begin{array}{cc}\\mathbf{W}\_{0} & \\mathbf{O} \\\\\\mathbf{O} & \\mathbf{U}\_0\\end{array}\\right)\n%\n\\left(\\begin{array}{c}\\mathbf{I}\_{M/2} \\\\ \n\\mathbf{O} \\\\\n\\mathbf{I}\_{M/2} \\\\\n\\mathbf{O}\n\\end{array}\\right)\\in\\mathbb{R}^{P\\times M}$,$\\mathbf{V}\_n^{\\{d\\}}=\\left(\\begin{array}{cc}\\mathbf{I}\_{P/2} & \\mathbf{O} \\\\\\mathbf{O} & \\mathbf{U}\_n^{\\{d\\}}\\end{array}\\right)\\in\\mathbb{R}^{P\\times P}, d\\in\\{\\mathrm{v},\\mathrm{h}\\}$, $\\mathbf{W}\_0, \\mathbf{U}\_0,\\mathbf{U}\_n^{\\{d\\}}\\in\\mathbb{R}^{P/2\\times P/2}$は直交行列
%[text] - $\\mathbf{Q}(z)=\\mathbf{B}\_{P}\\left(\\begin{array}{cc} \\mathbf{I}\_{P/2} &  \\mathbf{O} \\\\ \\mathbf{O} &  z^{-1}\\mathbf{I}\_{P/2}\\end{array}\\right)\\mathbf{B}\_{P}$, $\\bar{\\mathbf{Q}}(z)=\\mathbf{B}\_{P}\\left(\\begin{array}{cc} z\\mathbf{I}\_{P/2} &  \\mathbf{O} \\\\ \\mathbf{O} &  \\mathbf{I}\_{P/2}\\end{array}\\right)\\mathbf{B}\_{P}$, $\\mathbf{B}\_{P}=\\frac{1}{\\sqrt{2}}\\left(\\begin{array}{cc} \\mathbf{I}\_{P/2} &  \\mathbf{I}\_{P/2} \\\\ \\mathbf{I}\_{P/2} &  -\\mathbf{I}\_{P/2}\\end{array}\\right)$ \
%[text] 【References】 
%[text] - [Overview of Filter Banks - MATLAB & Simulink - MathWorks 日本](https://jp.mathworks.com/help/dsp/ug/overview-of-filter-banks.html)
%[text] - MATLAB SaivDr Package: [https://github.com/msiplab/SaivDr](https://github.com/msiplab/SaivDr)
%[text] - S. Muramatsu, K. Furuya and N. Yuki, "Multidimensional Nonseparable Oversampled Lapped Transforms: Theory and Design," in IEEE Transactions on Signal Processing, vol. 65, no. 5, pp. 1251-1264, 1 March1, 2017, doi: 10.1109/TSP.2016.2633240.
%[text] - S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks," in IEEE Transactions on Image Processing, vol. 21, no. 4, pp. 2314-2318, April 2012, doi: 10.1109/TIP.2011.2181527.
%[text] - S. Muramatsu, M. Ishii and Z. Chen, "Efficient parameter optimization for example-based design of nonseparable oversampled lapped transform," 2016 IEEE International Conference on Image Processing (ICIP), Phoenix, AZ, 2016, pp. 3618-3622, doi: 10.1109/ICIP.2016.7533034.
%[text] - Furuya, K., Hara, S., Seino, K., & Muramatsu, S. (2016). Boundary operation of 2D non-separable oversampled lapped transforms. *APSIPA Transactions on Signal and Information Processing, 5*, E9. doi:10.1017/ATSIP.2016.3. \
%[text] ### 2次元画像の階層的分析
%[text] $R\_M^P(\\tau)$ をツリーレベル $\\tau$の階層構造フィルタバンクの冗長度とすると、
%[text]  $R\_M^P(\\tau)=\\left\\{\\begin{array}{ll} (P-1)\\tau + 1, & M=1, \\\\ \\frac{P-1}{M-1}-\\frac{P-M}{(M-1)M^\\tau}, & M\\geq 2.\\end{array}\\right.$
%[text] となる．
%[text] #### 
%[text] #### 構成パラメータ設定
%%{
% Decimation factor (Strides)
decFactor = [2 2]; % [μv μh] 

% Number of channels ( sum(nChannels) >= prod(decFactors) )
nChannels = [4 4]; % [Ps Pa] (Ps=Pa)

% Number of tree levels
nLevels = 4; 

% Polyphase Order
ppOrder = [4 4]; 
%%}

%{
% Decimation factor (Strides)
decFactor =  [4 4]; % [μv μh] 

% Number of channels ( sum(nChannels) >= prod(decFactors) )
nChannels = [13 13]; % [Ps Pa] (Ps=Pa)

% Number of tree levels
nLevels = 2; 

% Polyphase Order
ppOrder = [2 2];
%}

%{
% Decimation factor (Strides)
decFactor =  [8 8]; % [μv μh] 

% Number of channels ( sum(nChannels) >= prod(decFactors) )
nChannels = [53 53]; % [Ps Pa] (Ps=Pa)

% Number of tree levels
nLevels = 1; 

% Polyphase Order
ppOrder = [2 2];
%}

% Redundancy
P = sum(nChannels);
M = prod(decFactor);
redundancyNsolt = ...
    (prod(decFactor)==1)*((P-1)*nLevels+1) + ...
    (prod(decFactor)>1)*((P-1)/(M-1)-(P-M)/((M-1)*M^nLevels))
assert(redundancyNsolt<redundancyRatio)

%[text] $L\_\\mathrm{v}\\times L\_\\mathrm{h}=\\left(\\mu\_\\mathrm{v}^{\\tau}+{\\nu}\_\\mathrm{v}\\frac{\\mu\_\\mathrm{v}(\\mu\_\\mathrm{v}^{\\tau}-1)}{\\mu\_\\mathrm{v}-1}\\right) \\times\\left(\\mu\_\\mathrm{h}^{\\tau}+\\nu\_\\mathrm{h}\\frac{\\mu\_\\mathrm{h}(\\mu\_\\mathrm{h}^{\\tau}-1)}{\\mu\_\\mathrm{h}-1}\\right)$ 
% Filter size [ Ly Lx ]
maxDecFactor = decFactor.^nLevels;
szFilters = maxDecFactor + ppOrder.*decFactor.*(maxDecFactor-1)./(decFactor-1)

% Patch size for training
szPatchTrn = maxDecFactor.*ceil(szFilters./maxDecFactor) % > [ Ly Lx ]
%szPatchTrn = 2.^nextpow2(szFilters) % > [ Ly Lx ]
assert(all(szPatchTrn>szFilters))

% Number of patchs per image
nSubImgs = floor(nPatches*prod(szBlk./szPatchTrn))
assert(nSubImgs > 0)

% No DC-leakage
noDcLeakage = true %[control:checkbox:4495]{"position":[15,19]}
%%
%[text] #### 辞書の設定
if exist("../data/"+nsoltDic+".mat","file")
    S = load("../data/"+nsoltDic);
    analysisnet = S.analysisnet;
    synthesisnet = S.synthesisnet;
    nLevels_ = extractnumlevels(analysisnet);
    decFactor_ = extractdecfactor(analysisnet);
    nChannels_ = extractnumchannels(analysisnet);

    assert(nLevels==nLevels_)
    assert(all(decFactor==decFactor_))
    assert(all(nChannels==nChannels_))
else
    % Number of iterations
    nItersNsolt = 10;

    % Standard deviation of initial angles
    stdInitAng = 1e-1; %pi/6;

    % Mini batch size
    miniBatchSize = 10;

    % Number of Epochs (1 Epoch = nSubImgs/miniBachSize iterlations)
    maxEpochs = 30;

    % Number of iterations
    maxIters = nSubImgs/miniBatchSize * maxEpochs

    % Training options
    opts = trainingOptions('sgdm', ... % Stochastic gradient descent w/ momentum
        ...'Momentum', 0.9000,...
        'InitialLearnRate',5.0e-03,...
        ...'LearnRateScheduleSettings','none',...
        'L2Regularization',0.0, ... 1.0e-04,... 
        ...'GradientThresholdMethod','l2norm',...
        ...'GradientThreshold',Inf,...
        'MaxEpochs',maxEpochs,...30,...
        'MiniBatchSize',miniBatchSize,...128,...
        'Verbose',1,...
        ...'VerboseFrequency',50,...
        ...'ValidationData',[],...
        ...'ValidationFrequency',50,...
        ...'ValidationPatience',Inf,...
        ...'Shuffle','once',...
        ...'CheckpointPath','',...
        ...'ExecutionEnvironment','auto',...
        ...'WorkerLoad',[],...
        ...'OutputFcn',[],...
        'Plots','none',...'training-progress',...
        ...'SequenceLength','longest',...
        ...'SequencePaddingValue',0,...
        ...'SequencePaddingDirection','right',...
        ...'DispatchInBackground',0,...
        'ResetInputNormalization',0);...1
%[text] #### 層構造の構築
    import saivdr.dcnn.*
    analysislgraph = fcn_creatensoltlgraph2d([],...
        'InputSize',szPatchTrn,...
        'NumberOfChannels',nChannels,...
        'DecimationFactor',decFactor,...
        'PolyPhaseOrder',ppOrder,...
        'NumberOfLevels',nLevels,...
        'NumberOfVanishingMoments',noDcLeakage,...
        'Mode','Analyzer');
    synthesislgraph = fcn_creatensoltlgraph2d([],...
        'InputSize',szPatchTrn,...
        'NumberOfChannels',nChannels,...
        'DecimationFactor',decFactor,...
        'PolyPhaseOrder',ppOrder,...
        'NumberOfLevels',nLevels,...
        'NumberOfVanishingMoments',noDcLeakage,...
        'Mode','Synthesizer');

    figure
    subplot(1,2,1)
    plot(analysislgraph)
    title('Analysis NSOLT')
    subplot(1,2,2)
    plot(synthesislgraph)
    title('Synthesis NSOLT')

    % Construction of deep learning network.
    synthesisnet = dlnetwork(synthesislgraph);

    % Initialize
    nLearnables = height(synthesisnet.Learnables);
    for iLearnable = 1:nLearnables
        if synthesisnet.Learnables.Parameter(iLearnable)=="Angles"
            layerName = synthesisnet.Learnables.Layer(iLearnable);
            synthesisnet.Learnables.Value(iLearnable) = ...
                cellfun(@(x) x+stdInitAng*randn(size(x)), ...
                synthesisnet.Learnables.Value(iLearnable),'UniformOutput',false);
        end
    end

    % Copy the synthesizer's parameters to the analyzer
    synthesislgraph = layerGraph(synthesisnet);
    analysislgraph = fcn_cpparamssyn2ana(analysislgraph,synthesislgraph);
    analysisnet = dlnetwork(analysislgraph);
%[text] #### 随伴関係（完全再構成）の確認
%[text] NSOLTはパーセバルタイト性を満たす．
    nOutputs = nLevels+1;
    x = rand(szPatchTrn,'single');
    s = cell(1,nOutputs);
    dlx = dlarray(x,'SSCB'); % Deep learning array (SSCB: Spatial,Spatial,Channel,Batch)
    [s{1:nOutputs}] = analysisnet.predict(dlx);
    dly = synthesisnet.predict(s{:});
    display("MSE: " + num2str(mse(dlx,dly)))
%[text] #### 要素画像の初期状態
    import saivdr.dcnn.*
    figure
    atomicimshow(synthesisnet,[],2^(nLevels-1))
    title('Atomic images of initial NSOLT')
%[text] ### 訓練画像の準備
%[text] 画像データストアからランダムにパッチを抽出
%[text] PCAに合わせて予め零平均化したデータで学習
    imds = imageDatastore(file_yorg,"ReadFcn",@(x) meansubtract(im2single(imread(x))));
    patchds = randomPatchExtractionDatastore(imds,imds,szPatchTrn,'PatchesPerImage',nSubImgs);
    figure
    minibatch = preview(patchds);
    responses = minibatch.ResponseImage;
    responses = cellfun(@(x) x + 0.5,responses,'UniformOutput',false);
    figure
    montage(responses,'Size',[2 4]);
    drawnow
%[text] ### 畳み込み辞書学習
%[text] #### 問題設定:
%[text]  $\\{\\hat{\\mathbf{\\theta}},\\{ \\hat{\\mathbf{x}}\_n \\}\\}=\\arg\\min\_{\\{\\mathbf{\\theta},\\{\\mathbf{x}\_n\\}\\}}\\frac{1}{2S}\\sum\_{n=1}^{S}\\|\\mathbf{y}\_n-\\mathbf{D}\_{\\mathbf{\\theta}}\\hat{\\mathbf{x}}\_n\\|\_2^2,\\ \\quad\\mathrm{s.t.}\\ \\forall n, \\|\\mathbf{x}\_n\\|\_0\\leq K,&dollar&;$
%[text] ただし， $\\mathbf{D}\_{\\mathbf{\\theta}}$は設計パラメータベクトル $\\mathbf{\\theta}}$をもつ畳み込み辞書．
%[text] 
%[text] #### アルゴリズム:
%[text] スパース近似ステップと辞書更新ステップを繰返す．
%[text] - スパース近似ステップ \
%[text]  $\\hat{\\mathbf{x}}\_n=\\arg\\min\_{\\mathbf{x}\_n}\\frac{1}{2} \\|\\mathbf{y}\_n-\\hat{\\mathbf{D}}\\mathbf{x}\_n\\|\_2^2\\ \\quad \\mathrm{s.t.}\\ \\|\\mathbf{x}\_n\\|\_0\\leq K$
%[text] - 辞書更新ステップ \
%[text]  $\\hat{\\mathbf{\\theta}}=\\arg\\min\_{\\mathbf{\\theta}}\\frac{1}{2S}\\sum\_{n=1}^{S}\\|\\mathbf{y}\_n-\\mathbf{D}\_{\\mathbf{\\theta}}\\hat{\\mathbf{x}}\_n\\|\_2^2$
%[text]  $\\hat{\\mathbf{D}}=\\mathbf{D}\_{\\hat{\\mathbf{\\theta}}$
%[text] #### 採用するスパース近似と辞書更新の手法:
%[text] - スパース近似：（正規化なし）繰返しハード閾値処理(IHT)
%[text] - 辞書更新： モーメンタム付き確率的勾配降下法(SGD) \
    % Check if IHT works for dlarray
    %x = dlarray(randn(szPatchTrn,'single'),'SSCB');
    %[y,coefs{1:nOutputs}] = iht(x,analysisnet,synthesisnet,sparsityRatio);
%[text] #### 辞書学習の繰返し計算
    import saivdr.dcnn.*
    %profile on
    for iIter = 1:nItersNsolt

        % Sparse approximation (Applied to produce an object of TransformedDatastore)
        coefimgds = transform(patchds, @(x) iht4patchds(x,analysisnet,synthesisnet,sparsityRatio));

        % Synthesis dictionary update
        trainlgraph = synthesislgraph.replaceLayer('Lv1_Out',...
            regressionLayer('Name','Lv1_Out'));
        trainednet = trainNetwork(coefimgds,trainlgraph,opts);

        % Analysis dictionary update (Copy parameters from synthesizer to analyzer)
        trainedlgraph = layerGraph(trainednet);
        analysislgraph = fcn_cpparamssyn2ana(analysislgraph,trainedlgraph);
        analysisnet = dlnetwork(analysislgraph);

        % Check the adjoint relation (perfect reconstruction)
        checkadjointrelation(analysislgraph,trainedlgraph,nLevels,szPatchTrn);

        % Replace layer
        synthesislgraph = trainedlgraph.replaceLayer('Lv1_Out',...
            nsoltIdentityLayer('Name','Lv1_Out'));
        synthesisnet = dlnetwork(synthesislgraph);

    end
    %profile off
    %profile viewer
%[text] #### 訓練ネットワークの保存
    import saivdr.dcnn.*
    synthesislgraph = layerGraph(synthesisnet);
    analysislgraph = fcn_cpparamssyn2ana(analysislgraph,synthesislgraph);
    analysisnet = dlnetwork(analysislgraph);
    save(sprintf('../results/nsoltdictionary_%s',datetime('now','Format','yyyyMMddHHmmssSSS')),'analysisnet','synthesisnet','nLevels')
end
%%
analysislgraph = layerGraph(analysisnet);
synthesislgraph = layerGraph(synthesisnet);

figure
subplot(1,2,1)
plot(analysislgraph)
title('Analysis NSOLT')
subplot(1,2,2)
plot(synthesislgraph)
title('Synthesis NSOLT')
%[text] #### 要素画像の表示
import saivdr.dcnn.*

figure

atomicimshow(synthesisnet,[],2^(nLevels-1))
title('Atomic images of trained NSOLT')
%[text] ### 推論用NSOLTネットワークの構築
%[text] dlnetworkは（DAGNetworkと異なり）入力層のサイズに縛られず任意の画像サイズで順伝播できるため、学習時のanalysisnet/synthesisnetをそのまま推論に用いる。これによりassembleNetwork用の層差し替えが不要になり、predict呼び出しも約1.6倍高速化する。
%[text] #### 随伴関係（完全再構成）の確認
%[text] NSOLTはパーセバルタイト性を満たす．
u = rand(szOrg,'single');
dlu = dlarray(u,'SSCB');
[s{1:nLevels+1}] = analysisnet.predict(dlu);
dlv = synthesisnet.predict(s{1:nLevels+1});
assert(mse(extractdata(dlv),u)<1e-9)
%[text] #### NSOLTによる合成処理とその随伴処理の定義
%[text] レベルごとに単一段（1レベル分）の要素画像（等価FIRフィルタ）を学習済みネットワークから抽出し、レベルの深さ分だけ`dlconv`/`dltranspconv`を再帰的に適用する階層的（カスケード）方式で高速化する。全レベルの係数を単一の畳み込みに平坦化する方式は各レベルの解像度差（decFactor^level）を無視してしまい多レベルでは完全再構成を満たせないため、必ずレベルごとに畳み込みを重ねる。
import saivdr.dcnn.*
maxDecFactor1 = decFactor; % 単一レベル分の等価フィルタ長を計算（nLevels=1相当）
szFilters1 = maxDecFactor1 + ppOrder.*decFactor.*(maxDecFactor1-1)./(decFactor-1);
protolgraph = fcn_creatensoltlgraph2d([],...
    'InputSize',szFilters1,...
    'NumberOfChannels',nChannels,...
    'DecimationFactor',decFactor,...
    'PolyPhaseOrder',ppOrder,...
    'NumberOfLevels',1,...
    'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Synthesizer');
protonetTemplate = dlnetwork(protolgraph);
nsoltconfig.nLevels = nLevels;
nsoltconfig.nChsPerLv = sum(nChannels);
nsoltconfig.decFactor = decFactor;
nsoltconfig.szOrg = szOrg;
nsoltconfig.padSz = (szFilters1 - decFactor)/2;
nsoltconfig.W = cell(1,nLevels);
for iLv = 1:nLevels
    protoLv = cplevelangles(synthesisnet,protonetTemplate,iLv);
    nsoltconfig.W{iLv} = single(getatomicimages(protoLv,szFilters1,1));
end
syn_nsolt = @(x) synthesisnsolt(x,nsoltconfig);
adj_nsolt = @(y) analysisnsolt(y,nsoltconfig);
%[text] #### 随伴関係の確認
x = adj_nsolt(y);
v = randn(size(x));
u = syn_nsolt(v);
assert(abs(dot(y(:),u(:))-dot(x(:),v(:)))<1e-3)
%%
%[text] ## 繰返しハード閾値処理(IHT)によるスパース近似の比較
%[text] #### 辞書の準備
blkdctwon  = { syn_blkdct,  adj_blkdct,  "Block DCTwoN", false };
blkdct  = { syn_blkdct,  adj_blkdct,  "Block DCT", true };
blkpcawon  = { syn_blkpca,  adj_blkpca,  "Block PCAwoN", false };
blkpca  = { syn_blkpca,  adj_blkpca,  "Block PCA", true };
blkrica = { syn_blkrica, adj_blkrica, "Block RICA", true };
blkksvd = { syn_blkksvd, adj_blkksvd, "Block K-SVD", true };
nsoltwon   = { syn_nsolt,   adj_nsolt,   "NSOLTwoN", false };
nsolt = { syn_nsolt,   adj_nsolt,   "NSOLT", true };
dicset  = { blkdctwon, blkdct, blkpcawon, blkpca, blkrica, blkksvd, nsoltwon, nsolt };
nDics   = length(dicset);
%[text] #### IHT
%[text]  $\\mathbf{x}^{(t+1)}\\leftarrow \\mathcal{H}\_{BK}\\left(\\mathbf{x}^{(t)}+\\mu^{(t)}\\hat{\\mathbf{D}}^\\textsf{T}\\left(\\mathbf{y}-\\hat{\\mathbf{D}}\\mathbf{x}^{(t)}\\right)\\right)$
%[text]  $t\\leftarrow t+1$
%[text] -  T. Blumensath and M. E. Davies, "Normalized Iterative Hard Thresholding: Guaranteed Stability and Performance," in IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 2, pp. 298-309, April 2010, doi: 10.1109/JSTSP.2010.2042411. \
nItersIht = 2000;

% 平均値を引いた画像を用意（近似後に平均値を加算）
ymean = mean(yorg,"all");
y = yorg - ymean;
% 準備
c = 1e-3;
kappa = 1.1/(1-c);
nCoefs = floor(sparsityRatio*prod(szOrg));
psnrs = zeros(nItersIht,nDics);
ssims = zeros(nItersIht,nDics);
yaprxs = cell(1,nDics);
% 繰り返し処理
for iDic = 1:nDics
    dic_ = dicset{iDic};
    synproc = dic_{1};
    adjproc = dic_{2};
    dicname = dic_{3};
    isStepSizeNormalized = dic_{4};
    % IHT
    display(dicname)
    s = adjproc(y); % D^Ty
    xt = zeros(size(s),'like',s); % x1 = 0;
    yaprx_ = zeros(size(y),'like',y); % synproc(x1)=0 に相当。次段のgt計算で
    % 前反復のsynproc(xt)結果を再利用し、毎反復2回計算していたsynprocの呼出しを1回に減らす
    if isStepSizeNormalized % 正規化あり
        suppt = find(hardthresh(s,nCoefs)); % Γ1 = supp(H_K(D^Ty))
        maskt = (abs(s)~=0);
    end
    for iIter=1:nItersIht
        % Gradient descent
        gt = adjproc(y-yaprx_); % g = D^T(y-Dxn)
        if ~isStepSizeNormalized % 正規化なし
            mu = (1-c);
            xtp1 = hardthresh(xt+mu*gt,nCoefs);
        else % 正規化あり
            ggt = gt(suppt); % g_Γn
            ugt = synproc(maskt.*gt); % D_Γn^T g_Γn
            mu = (ggt.'*ggt)/(ugt(:).'*ugt(:));
            ttp1 = hardthresh(xt+mu*gt,nCoefs); % ~xn+1 = H_K(xn + μn gn)
            supptp1 = find(ttp1); % Γn+1 = supp(~xn+1)
            if length(supptp1)==length(suppt) && all(supptp1==suppt)
                xtp1 = ttp1; % xn+1 = ~xn+1
            else
                dxt = ttp1-xt; % ~xn+1 - xn
                omega = (1-c)*(norm(dxt,'fro')/norm(synproc(dxt),'fro'))^2;
                if mu <= omega
                    xtp1 = ttp1; % xn+1 = ~xn+1
                else
                    while mu > omega
                        mu = mu/(kappa*(1-c));
                        ttp1 = hardthresh(xt+mu*gt,nCoefs); % ~xn+1 = H_K(xn + μn gn)
                        dxt = ttp1-xt; % ~xn+1 - xn
                        omega = (1-c)*(norm(dxt,'fro')/norm(synproc(dxt),'fro'))^2;
                    end
                    supptp1 = find(ttp1);  % Γn+1 = supp(~xn+1)
                    xtp1 = ttp1; % xn+1 = ~xn+1
                end
            end
            % Update
            suppt = supptp1;
            maskt = zeros(size(maskt),'like',maskt);
            maskt(suppt) = 1;
        end
        xt = xtp1;
        % Monitoring
        checkSparsity = nnz(xt)/prod(szOrg)<=sparsityRatio;
        assert(checkSparsity)
        yaprx_ = synproc(xt); % 次反復のgt計算でも再利用するのでキャッシュしておく
        psnr_ = psnr(cast(yaprx_,'like',y),y);
        ssim_ = ssim(cast(yaprx_,'like',y),y);
        psnrs(iIter,iDic) = psnr_;
        ssims(iIter,iDic) = ssim_;
        %fprintf("IHT(%d) PSNR: %6.4f\n",iIter,psnr_);
    end
    yaprxs{iDic} = yaprx_ + ymean;
end
%%
%[text] ## 近似結果の表示
dicnames = [blkdctwon{3},blkdct{3},blkpcawon{3},blkpca{3},blkrica{3},blkksvd{3},nsoltwon{3},nsolt{3}];
psnrtbl = array2table(psnrs,'VariableNames',dicnames);
psnrtbl = horzcat(table((1:nItersIht).','VariableNames',"Iterations"),psnrtbl);
ssimtbl = array2table(ssims,'VariableNames',dicnames);
ssimtbl = horzcat(table((1:nItersIht).','VariableNames',"Iterations"),ssimtbl);

% PSNR のグラフ
figure
plot(psnrtbl,"Iterations",dicnames,'LineWidth',2)
ylabel('PSNR [dB]')
legend('Location','best')
% SSIM のグラフ
figure
plot(ssimtbl,"Iterations",dicnames,'LineWidth',2)
ylabel('SSIM')
legend('Location','best')


%%
% 原画像の表示
figure
tiledlayout(2,ceil((nDics+1)/2))
nexttile
imshow(yorg)
title("Original image")
% 近似画像の表示
for idx = 1:nDics
    yaprx = yaprxs{idx};
    dicname = dicnames(idx)
    file_yaprx = "../results/yaprx_" + replace(lower(dicname),' ','_') +".png";
    imwrite(yaprx,file_yaprx)
    %
    nexttile
    imshow(yaprxs{idx})
    title(dicname+" "+num2str(psnrs(end,idx))+" dB")
end
%%
%[text] ## 【関数定義】
%[text] #### レベルiLvの学習済み回転角を単一レベルのひな型ネットワークにコピー
function protoLv = cplevelangles(fullnet,protoTemplate,iLv)
% レベル毎に独立な回転角(Angles)を持つ多レベル網から、レベルiLv分の値だけを
% 単一レベル(nLevels=1)のひな型ネットワークにコピーする。
protoLv = protoTemplate;
prefixFrom = sprintf('Lv%d_',iLv);
for i = 1:height(fullnet.Learnables)
    name = fullnet.Learnables.Layer(i);
    if startsWith(name,prefixFrom)
        newname = "Lv1_"+extractAfter(name,prefixFrom);
        j = find(protoLv.Learnables.Layer==newname & ...
            protoLv.Learnables.Parameter==fullnet.Learnables.Parameter(i));
        if ~isempty(j)
            protoLv.Learnables.Value(j) = fullnet.Learnables.Value(i);
        end
    end
end
end
%[text] #### 単一レベル分の要素画像(atomic images)の抽出
function [atomicImages, mRows, mCols] = getatomicimages(synthesisnet, patchsize, scale)
% GETATOMICIMAGES 学習済みNSOLT合成網（単一レベル）から要素画像を計算する
% 各要素インパルス入力に対する合成網の応答を求めることで、単一レベル分の
% ラティス構造と等価な多チャネルFIRフィルタ（畳み込みカーネル）を構築する。
import saivdr.dcnn.*
if nargin < 3 || isempty(scale)
    scale = 1;
end
expfinallayer = '^Lv1_Cmp1+_V0~?$';
expidctlayer = '^Lv\d+_E0~?$';
nLayers = height(synthesisnet.Layers);
nLevels = 0;
nComponents = 1;
for iLayer = 1:nLayers
    layer = synthesisnet.Layers(iLayer);
    if ~isempty(regexp(layer.Name,expfinallayer,'once'))
        nChannels = layer.NumberOfChannels;
        decFactor = layer.DecimationFactor;
    end
    if ~isempty(regexp(layer.Name,expidctlayer,'once'))
        nLevels = nLevels + 1;
        if nLevels == 1
            nComponents = layer.NumInputs;
        end
    end
end
nChsPerLv = sum(nChannels);
nChsTotal = nLevels*(nChsPerLv-1)+1;
DIMENSION = 2;
MARGIN = 2;
if nargin < 2 || isempty(patchsize)
    estPpOrder = floor([1 1]*sqrt(nLayers/(DIMENSION*nLevels)));
    estKernelExt = decFactor.*(estPpOrder+1);
    for iLv = 2:nLevels
        estKernelExt = (estKernelExt-1).*(decFactor+1)+1;
    end
    maxDecFactor = decFactor.^nLevels;
    patchsize = (ceil(estKernelExt./maxDecFactor)+MARGIN).*maxDecFactor;
end
atomicImages = zeros([patchsize 1 nChsTotal],'single');
dls = cell(nLevels+1,1);
for iRevLv = nLevels:-1:1
    if iRevLv == nLevels
        dls{nLevels+1} = dlarray(zeros([patchsize./(decFactor.^nLevels) nComponents],'single'),'SSC');
        dls{nLevels} = dlarray(zeros([patchsize./(decFactor.^nLevels) nComponents*(nChsPerLv-1)],'single'),'SSC');
    else
        dls{iRevLv} = dlarray(zeros([patchsize./(decFactor.^iRevLv) nComponents*(nChsPerLv-1)],'single'),'SSC');
    end
end
idx = 1;
dld = dls;
dld{nLevels+1}(round(end/2),round(end/2),1:nComponents) = ones(1,1,nComponents);
atomicImages(:,:,1:nComponents,idx) = extractdata(synthesisnet.predict(dld{:}));
idx = idx+1;
for iRevLv = nLevels:-1:1
    for iAtom = 1:nChsPerLv-1
        dld = dls;
        for iCmp = 1:nComponents
            dld{iRevLv}(round(end/2),round(end/2),(iCmp-1)*(nChsPerLv-1)+iAtom) = 1;
        end
        atomicImages(:,:,1:nComponents,idx) = extractdata(synthesisnet.predict(dld{:}));
        idx = idx+1;
    end
end
atomicImages = scale * atomicImages;
mRows = 2^(nextpow2(sqrt(nChsTotal))-1);
mCols = ceil(nChsTotal/mRows);
end
%[text] #### 単一レベル分の合成畳み込み（転置畳み込み＋周期折返し）
function y = synthesisnsolt_conv(x, W_syn, decFactor, padSz, szSub)
x = cast(x,'like',W_syn);
x_3d = reshape(x, szSub);
x_dl = dlarray(x_3d, 'SSC');
bias_s = zeros(1,'like',W_syn);
y_full_dl = dltranspconv(x_dl, W_syn, bias_s, 'Stride', decFactor, 'Cropping', 0);
y_full = extractdata(y_full_dl); % [H_full W_full 1]
% 周期折返し（NSOLTの周期境界条件を再現）
p_H = padSz(1); p_W = padSz(2);
N_H = (szSub(1)-1)*decFactor(1) + size(W_syn,1) - 2*p_H;
N_W = (szSub(2)-1)*decFactor(2) + size(W_syn,2) - 2*p_W;
y = y_full(p_H+1:p_H+N_H, p_W+1:p_W+N_W, 1);
y(end-p_H+1:end,:) = y(end-p_H+1:end,:) + y_full(1:p_H, p_W+1:p_W+N_W, 1);
y(1:p_H,:)         = y(1:p_H,:)         + y_full(p_H+N_H+1:end, p_W+1:p_W+N_W, 1);
y(:,end-p_W+1:end) = y(:,end-p_W+1:end) + y_full(p_H+1:p_H+N_H, 1:p_W, 1);
y(:,1:p_W)         = y(:,1:p_W)         + y_full(p_H+1:p_H+N_H, p_W+N_W+1:end, 1);
y(end-p_H+1:end,end-p_W+1:end) = y(end-p_H+1:end,end-p_W+1:end) + y_full(1:p_H, 1:p_W, 1);
y(end-p_H+1:end,1:p_W)         = y(end-p_H+1:end,1:p_W)         + y_full(1:p_H, p_W+N_W+1:end, 1);
y(1:p_H,end-p_W+1:end)         = y(1:p_H,end-p_W+1:end)         + y_full(p_H+N_H+1:end, 1:p_W, 1);
y(1:p_H,1:p_W)                 = y(1:p_H,1:p_W)                 + y_full(p_H+N_H+1:end, p_W+N_W+1:end, 1);
end
%[text] #### 単一レベル分の分析畳み込み（周期拡張＋畳み込み）
function x = analysisnsolt_conv(y, W_syn, decFactor, padSz)
y = cast(y,'like',W_syn);
if ismatrix(y); y = reshape(y,[size(y,1) size(y,2) 1]); end
p_H = padSz(1); p_W = padSz(2);
y_pad = padarray(y, [p_H p_W 0], 'circular', 'both'); % 周期拡張
y_dl = dlarray(y_pad, 'SSC');
bias_a = zeros(size(W_syn,4),1,'like',W_syn);
x_dl = dlconv(y_dl, W_syn, bias_a, 'Stride', decFactor, 'Padding', 0);
x = extractdata(x_dl);
x = x(:);
end
%[text] #### NSOLT合成処理関数（レベルごとの畳み込みを再帰的に適用する階層的方式）
function y = synthesisnsolt(x,config)
nLevels = config.nLevels;
nChsPerLv = config.nChsPerLv;
decFactor = config.decFactor;
padSz = config.padSz;
szOrg = config.szOrg;
nAc = nChsPerLv - 1;
sidx = 1;
acParts = cell(1,nLevels);
for iLv = 1:nLevels
    szSub = szOrg./(decFactor.^iLv);
    n = prod(szSub)*nAc;
    acParts{iLv} = reshape(x(sidx:sidx+n-1),[szSub nAc]);
    sidx = sidx+n;
end
szDc = szOrg./(decFactor.^nLevels);
dc = reshape(x(sidx:sidx+prod(szDc)-1),szDc);
for iLv = nLevels:-1:1
    szSub = szOrg./(decFactor.^iLv);
    combined = cat(3,dc,acParts{iLv}); % [DC, AC_1..AC_nAc] の順で要素画像の並びと一致させる
    dc = synthesisnsolt_conv(combined(:),config.W{iLv},decFactor,padSz,[szSub nChsPerLv]);
end
y = dc;
end

%[text] #### NSOLT分析処理関数（レベルごとの畳み込みを再帰的に適用する階層的方式）
function x = analysisnsolt(y,config)
nLevels = config.nLevels;
nChsPerLv = config.nChsPerLv;
decFactor = config.decFactor;
padSz = config.padSz;
dc = y;
acParts = cell(1,nLevels);
for iLv = 1:nLevels
    combined = analysisnsolt_conv(dc,config.W{iLv},decFactor,padSz);
    szSub = size(dc)./decFactor;
    combined = reshape(combined,[szSub nChsPerLv]);
    dc = combined(:,:,1);
    acParts{iLv} = combined(:,:,2:end);
end
parts = cell(1,nLevels+1);
for iLv = 1:nLevels
    parts{iLv} = acParts{iLv}(:);
end
parts{nLevels+1} = dc(:);
x = cat(1,parts{:});
end
%[text] #### ハード閾値処理
function y = hardthresh(x,K)
v = sort(abs(x(:)),'descend');
thk = v(K);
y = (abs(x)>thk).*x;
end
%[text] #### 深層学習配列に対する繰返しハード閾値処理(IHT)のバッチ処理
function newdata = iht4patchds(oldtbl,analyzer,synthesizer,sparsityRatio)
% IHT for InputImage in randomPatchExtractionDatastore
%
nInputs = length(synthesizer.InputNames);

% Apply IHT process for every input patch
restbl = removevars(oldtbl,'InputImage');
dlv = dlarray(cat(4,oldtbl.InputImage{:}),'SSCB');
[~,dlcoefs{1:nInputs}] = iht4dlarray(dlv,analyzer,synthesizer,sparsityRatio);
coefs = cellfun(@(x) permute(num2cell(extractdata(x),1:3),[4 1 2 3]),dlcoefs,'UniformOutput',false);
%
nImgs = length(oldtbl.InputImage);
coefarray = cell(nImgs,nInputs);
for iImg = 1:nImgs
    for iInput = 1:nInputs
        coefarray{iImg,iInput} = coefs{iInput}{iImg};
    end
end
% Output as a cell in order to make multiple-input datastore
newdata = [ coefarray table2cell(restbl) ];
end
%[text] #### 深層学習配列に対する繰返しハード閾値処理(IHT)
function [dly,varargout] = iht4dlarray(dlx,analyzer,synthesizer,sparsityRatio)
% IHT Iterative hard thresholding
%
nInputs = length(synthesizer.InputNames);
szBatch = size(dlx,4);

% Iterative hard thresholding w/o normalization
% (A Parseval tight frame is assumed)
gamma = (1.-1e-3);
nIters = 10;
nCoefs = floor(sparsityRatio*numel(dlx(:,:,:,1)));
[dlcoefs{1:nInputs}] = analyzer.predict(dlarray(zeros(size(dlx),'like',dlx),'SSCB'));
% IHT
for iter=1:nIters
    % Gradient descent
    dly = synthesizer.predict(dlcoefs{1:nInputs});
    [grad{1:nInputs}] = analyzer.predict(dlx-dly);
    dlcoefs = cellfun(@(x,y) x+gamma*y,dlcoefs,grad,'UniformOutput',false);
    % Hard thresholding
    coefvecs = cellfun(@(x) extractdata(reshape(x,[],szBatch)),dlcoefs,'UniformOutput',false);
    srtdabscoefs = sort(abs(cell2mat(coefvecs.')),1,'descend');
    thk = reshape(srtdabscoefs(nCoefs,:),1,1,1,szBatch);
    dlcoefs = cellfun(@(x) (abs(x)>thk).*x,dlcoefs,'UniformOutput',false);
    % Monitoring
    %checkSparsity =...
    %nnz(srtdabscoefs>srtdabscoefs(nCoefs,:))/numel(dlx)<=sparsityRatio;
    %assert(checkSparsity)
    %fprintf("IHT(%d) MSE: %6.4f\n",iter,mse(dlx,dly));
end
varargout = dlcoefs;
end
%[text] #### NSOLTネットワークの随伴関係の確認
function checkadjointrelation(analysislgraph,synthesislgraph,nLevels,szInput)
import saivdr.dcnn.*
x = rand(szInput,'single');
% Assemble analyzer
analysislgraph4predict = analysislgraph;
for iLayer = 1:length(analysislgraph4predict.Layers)
    layer = analysislgraph4predict.Layers(iLayer);
    if contains(layer.Name,"Lv"+nLevels+"_DcOut") || ...
            ~isempty(regexp(layer.Name,'^Lv\d+_AcOut','once'))
        analysislgraph4predict = analysislgraph4predict.replaceLayer(layer.Name,...
            regressionLayer('Name',layer.Name));
    end
end
analysisnet4predict = assembleNetwork(analysislgraph4predict);

% Assemble synthesizer
synthesislgraph4predict = synthesislgraph;
synthesisnet4predict = assembleNetwork(synthesislgraph4predict);

% Analysis and synthesis process
[s{1:nLevels+1}] = analysisnet4predict.predict(x);
if isvector(s{end-1})
    s{end-1} = permute(s{end-1},[1,3,2]);
end
y = synthesisnet4predict.predict(s{:});

% Evaluation
display("MSE: " + num2str(mse(x,y)))
end
%[text] #### 直交マッチング追跡関数の定義
function x = omp(y,Phi,nCoefs)
% Initializaton
nAtoms = size(Phi,2);
normsq = sum(Phi.^2,1).'; % ||dm||^2, m=1,...,nAtoms (constant over the loop)
x = zeros(nAtoms,1);
r = y;
supp = false(nAtoms,1);
k = 0;
while k < nCoefs
    % Matching process (vectorized over all atoms at once)
    g = Phi.'*r; % γm=<dm,r>
    a = g./normsq; % Normalize αm=γm/||dm||^2
    e = (r.'*r) - g.*a; % <r-dm/||dm||^2,r>
    e(supp) = Inf; % Exclude atoms already in the support
    % Minimum value search (pursuit)
    [~,mmin] = min(e);
    % Update the support
    supp(mmin) = true;
    subPhi = Phi(:,supp);
    x(supp) = subPhi \ y; % Least squares via QR, faster than pinv
    % Residual
    r = y - Phi*x;
    % Update
    k = k + 1;
end
end
%[text] #### NSOLTネットワークからのツリーレベル情報の抽出
function nLevels = extractnumlevels(nsoltnet)
import saivdr.dcnn.*

% Extraction of information
expidctlayer = '^Lv\d+_E0~?$';
nLevels = 0;
nLayers = height(nsoltnet.Layers);
for iLayer = 1:nLayers
    layer = nsoltnet.Layers(iLayer);
    if ~isempty(regexp(layer.Name,expidctlayer,'once'))
        nLevels = nLevels + 1;
    end
end
end
%[text] #### NSOLTネットワークからのストライド情報の抽出
function decFactor = extractdecfactor(nsoltnet)
import saivdr.dcnn.*

% Extraction of information
expfinallayer = '^Lv1_Cmp1+_V0~?$';
nLayers = height(nsoltnet.Layers);
for iLayer = 1:nLayers
    layer = nsoltnet.Layers(iLayer);
    if ~isempty(regexp(layer.Name,expfinallayer,'once'))
        decFactor = layer.DecimationFactor;
    end
end
end
%[text] #### NSOLTネットワークからのチャネル数情報の抽出
function nChannels = extractnumchannels(nsoltnet)
import saivdr.dcnn.*

% Extraction of information
expfinallayer = '^Lv1_Cmp1+_V0~?$';
nLayers = height(nsoltnet.Layers);
for iLayer = 1:nLayers
    layer = nsoltnet.Layers(iLayer);
    if ~isempty(regexp(layer.Name,expfinallayer,'once'))
        nChannels = layer.NumberOfChannels;
    end
end
end
%[text] © Copyright, 2023-2026, Shogo MURAMATSU, All rights reserved.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":20.7}
%---
%[control:checkbox:6a42]
%   data: {"defaultValue":false,"label":"isCodegen","run":"Section"}
%---
%[control:checkbox:4495]
%   data: {"defaultValue":true,"label":"noDcLeakage","run":"Section"}
%---
%[output:5beb9c5c]
%   data: {"dataType":"text","outputData":{"text":"SaivDr-4.2.2.5 exits.\nSkip code generation\n","truncated":false}}
%---
