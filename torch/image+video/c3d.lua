local function c3d(nGPU)
   -- Create table describing C3D configuration
   local cfg = {64, 'M1', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}

   local features = nn.Sequential()
   do
      local iChannels = 3;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(nn.VolumetricMaxPooling(2,2,2,2,2,2):ceil())
         elseif v == 'M1' then
            features:add(nn.VolumetricMaxPooling(1,2,2,1,2,2):ceil())
         else
            local oChannels = v;
            features:add(nn.VolumetricConvolution(iChannels,oChannels,3,3,3,1,1,1,1,1,1))
            features:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end

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

   features:get(1).gradInput = nil

   local classifier = nn.Sequential()
   classifier:add(nn.View(512*1*4*4))
   classifier:add(nn.Linear(512*1*4*4, 4096))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 101)) -- UCF-101

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model, {3,16,112,112}
end

return c3d
