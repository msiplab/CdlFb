% EXPERIMENTAL_NSOLT_CONVIMPL 多レベルNSOLTの畳み込み実装（研究中・未完成）
%
% 目的:
%   NSOLT合成・分析処理を、ラティス層を毎回走査するdlnetwork.predictの代わりに
%   dlconv/dltranspconvによる単発の畳み込み演算で高速化すること。
%
% 現状のステータス: 未完成。完全再構成を厳密には満たさない（下記ベンチマーク参照）。
%   実運用の sample_cdlfb.m には組み込んでいない。安全に検証済みの高速化
%   （dlnetwork直接predict、yaprx_キャッシュ）のみ main ブランチに反映済み。
%
% 経緯・デバッグ履歴:
%   1. 素朴な実装（要素画像を1回だけ抽出し、全チャンネルを単一ストライドdecFactorで
%      畳み込む）は nLevels=1 でのみ正しい（完全再構成MSE~1e-15）。
%      nLevels>1では、各レベルの係数が実際には異なる解像度（decFactor^level）を
%      持つにもかかわらず、単一ストライドで平坦化してしまうため破綻する。
%      nLevels=4(本スクリプトの設定)でMSE~4.2e6、参照元example09_03.mの設定
%      (nLevels=3)でもMSE~16558と同様に破綻することを確認。
%      随伴関係チェックだけでは検出できない
%      （dlconv/dltranspconvは内容が誤っていても常に厳密な随伴ペアになるため）。
%   2. 各レベルの係数グループをそれぞれ自身の解像度（decFactor^level）で
%      dltranspconv/dlconvし、結果を足し合わせる方式に変更 → MSE~1466（3000倍改善）。
%   3. getatomicimagesに渡していた謎のスケール係数 2^(nLevels-1) を除去
%      （元コードでの用途はおそらくatomicimshowでの表示用輝度調整であり、
%      畳み込みカーネルとして使うなら不要／有害と判断） → MSE~0.038（さらに38000倍改善）。
%   4. 抽出窓(patchsize)を余分に大きくしても誤差は変化せず(0.038近辺で足踏み)。
%      → 単純な余白不足ではなく、レベル間の再帰的な境界条件の扱いに残る
%      構造的なバグが疑われる（各レベルを独立に周期折返しするのではなく、
%      再帰の各段で周期折返しを行う必要がある可能性など）。ここで探究を中断。
%
% 次にやるべきこと（着手する場合）:
%   - 周期折返しを「レベルごとに独立」ではなく「再帰の各段で」行うカスケード方式
%     （各レベルのAC係数とDC係数を単一レベル分だけ合成し、その出力を次段の入力に
%     使う、を再帰的に繰り返す）を実装し、MSEが機械精度(1e-9程度)まで下がるか検証する。
%   - あるいは、SaivDrパッケージのレイヤー構造を直接調べ、各レベルの単一レベル
%     プロトタイプフィルタ（V0/Vh/Vv角度）から直接FIRフィルタ行列を代数的に
%     構成する方法（ネットワーク実行によるインパルス応答の"探り読み"に依らない
%     方法）も検討に値する。
%
% 実行するには SaivDr パッケージが必要（setup.m 参照）。

clear
import saivdr.dcnn.*

decFactor = [2 2];
nChannels = [4 4];
nLevels = 4;
ppOrder = [4 4];
noDcLeakage = true;
szOrg = [192 256];
maxDecFactor = decFactor.^nLevels;
szFilters = maxDecFactor + ppOrder.*decFactor.*(maxDecFactor-1)./(decFactor-1);
szPatchTrn = maxDecFactor.*ceil(szFilters./maxDecFactor);

synthesislgraph = fcn_creatensoltlgraph2d([],...
    'InputSize',szPatchTrn,'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
    'PolyPhaseOrder',ppOrder,'NumberOfLevels',nLevels,'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Synthesizer');
analysislgraph = fcn_creatensoltlgraph2d([],...
    'InputSize',szPatchTrn,'NumberOfChannels',nChannels,'DecimationFactor',decFactor,...
    'PolyPhaseOrder',ppOrder,'NumberOfLevels',nLevels,'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Analyzer');
synthesisnet = dlnetwork(synthesislgraph);
analysisnet = dlnetwork(analysislgraph); %#ok<NASGU>

% ランダムな角度を与え、DCTの単純な初期状態から少し外す（テスト用）
nLearnables = height(synthesisnet.Learnables);
for iLearnable = 1:nLearnables
    if synthesisnet.Learnables.Parameter(iLearnable)=="Angles"
        synthesisnet.Learnables.Value(iLearnable) = ...
            cellfun(@(x) x+1e-1*randn(size(x)), synthesisnet.Learnables.Value(iLearnable),'UniformOutput',false);
    end
end
synthesislgraph = layerGraph(synthesisnet);
analysislgraph = fcn_cpparamssyn2ana(analysislgraph,synthesislgraph);
analysisnet = dlnetwork(analysislgraph); %#ok<NASGU>

nChsPerLv = sum(nChannels);
patchsize = maxDecFactor.*(ceil(szFilters./maxDecFactor)+4); % 余裕を持たせた抽出窓
W_syn = single(getatomicimages(synthesisnet,patchsize,1)); % スケール係数は使わない(=1)

u = rand(szOrg,'single');
y0 = rand(szOrg,'single');

% 完全再構成チェック: syn(analysis(u)) は u に一致するはず（パーセバルタイトフレーム）
xchk = analysisnsolt_multires(u, W_syn, decFactor, nLevels, nChsPerLv, patchsize);
vrec = synthesisnsolt_multires(xchk, W_syn, decFactor, nLevels, nChsPerLv, patchsize, szOrg);
fprintf('perfect-reconstruction MSE (multires conv) = %g  (目標: ~1e-9. 現状は未達)\n', ...
    mean((double(vrec(:))-double(u(:))).^2));

% 随伴関係チェック（この演算自体は常に近似的に成立するため、正しさの十分な検証にはならない）
xadj = analysisnsolt_multires(y0, W_syn, decFactor, nLevels, nChsPerLv, patchsize);
vadj = randn(size(xadj),'single');
uadj = synthesisnsolt_multires(vadj, W_syn, decFactor, nLevels, nChsPerLv, patchsize, szOrg);
lhs = dot(double(y0(:)),double(uadj(:)));
rhs = dot(double(xadj(:)),double(vadj(:)));
fprintf('adjoint check |lhs-rhs| = %g (lhs=%g, rhs=%g)\n', abs(lhs-rhs), lhs, rhs);

%% ---- レベル別ストライドで分離して畳み込み、結果を合算する（未完成） ----

function y = synthesisnsolt_multires(x, atoms, decFactor, nLevels, nChsPerLv, patchsize, szOrg)
levels = [nLevels, nLevels:-1:1]; % [DC, LvN AC, ..., Lv1 AC]
groupSizes = [1, repmat(nChsPerLv-1,1,nLevels)];
y = zeros(szOrg,'like',atoms);
idx = 1;
sidx = 1;
for g = 1:numel(levels)
    lv = levels(g);
    nAtomsInGroup = groupSizes(g);
    stride = decFactor.^lv; % そのグループの本来の解像度に応じたストライド
    szSub = szOrg./stride;
    nCoefsGroup = prod(szSub)*nAtomsInGroup;
    xg = x(sidx:sidx+nCoefsGroup-1);
    sidx = sidx + nCoefsGroup;
    Wg = atoms(:,:,1,idx:idx+nAtomsInGroup-1);
    idx = idx + nAtomsInGroup;
    padSz = (patchsize - stride)/2;
    yg = synthesisnsolt_conv(xg, Wg, stride, padSz, [szSub nAtomsInGroup]);
    y = y + yg;
end
end

function x = analysisnsolt_multires(y, atoms, decFactor, nLevels, nChsPerLv, patchsize)
levels = [nLevels, nLevels:-1:1];
groupSizes = [1, repmat(nChsPerLv-1,1,nLevels)];
xparts = cell(1,numel(levels));
idx = 1;
for g = 1:numel(levels)
    lv = levels(g);
    nAtomsInGroup = groupSizes(g);
    stride = decFactor.^lv;
    Wg = atoms(:,:,1,idx:idx+nAtomsInGroup-1);
    idx = idx + nAtomsInGroup;
    padSz = (patchsize - stride)/2;
    xparts{g} = analysisnsolt_conv(y, Wg, stride, padSz);
end
x = cat(1,xparts{:});
end

function y = synthesisnsolt_conv(x, W_syn, decFactor, padSz, szSub)
% 転置畳み込み(dltranspconv)＋周期折返し（単一レベル分。nLevels=1でのみ厳密に正しいことを確認済み）
x = cast(x,'like',W_syn);
x_3d = reshape(x, szSub);
x_dl = dlarray(x_3d, 'SSC');
bias_s = zeros(1,'like',W_syn);
y_full_dl = dltranspconv(x_dl, W_syn, bias_s, 'Stride', decFactor, 'Cropping', 0);
y_full = extractdata(y_full_dl);
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

function x = analysisnsolt_conv(y, W_syn, decFactor, padSz)
% 周期拡張＋畳み込み(dlconv)（単一レベル分）
y = cast(y,'like',W_syn);
if ismatrix(y); y = reshape(y,[size(y,1) size(y,2) 1]); end
p_H = padSz(1); p_W = padSz(2);
y_pad = padarray(y, [p_H p_W 0], 'circular', 'both');
y_dl = dlarray(y_pad, 'SSC');
bias_a = zeros(size(W_syn,4),1,'like',W_syn);
x_dl = dlconv(y_dl, W_syn, bias_a, 'Stride', decFactor, 'Padding', 0);
x = extractdata(x_dl);
x = x(:);
end

function [atomicImages, mRows, mCols] = getatomicimages(synthesisnet, patchsize, scale)
% GETATOMICIMAGES 学習済みNSOLT合成網から要素画像(atomic images)を計算する
% 各要素インパルス入力に対する合成網の応答を求めることで、ラティス構造全体と
% 等価な多チャネルFIRフィルタ（畳み込みカーネル）を構築する。
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
