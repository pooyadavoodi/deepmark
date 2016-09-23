local function alexnet(nGPU)
   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   require 'cudnn'
   local features = nn.Sequential()
   features:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.ReLU(true))
   features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.ReLU(true))
   features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local features_single = features
      features = nn.DataParallelTable(1)
      for i=1,nGPU do
          cutorch.setDevice(i)
          features:add(features_single:clone(), i)
      end
      features.gradInput = nil
      features.flattenParams = true
      cutorch.setDevice(1)
   end

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   local branch1
   branch1 = nn.Concat(2)
   local s = nn.Sequential()
   s:add(nn.Dropout(0.5))
   s:add(nn.Linear(256*6*6, 4096))
   s:add(nn.Threshold(0, 1e-6))
   branch1:add(s)
   classifier:add(branch1)

   local branch2
   branch2 = nn.Concat(2)
   local s = nn.Sequential()
   s:add(nn.Dropout(0.5))
   s:add(nn.Linear(4096, 4096))
   s:add(nn.Threshold(0, 1e-6))
   branch2:add(s)
   classifier:add(branch2)

   classifier:add(nn.Linear(4096, 1000))

   features:get(1).gradInput = nil

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model, {3,224,224}
end

return alexnet
