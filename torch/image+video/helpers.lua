local helpers = {}

local function setMultiGPU(opt, features)
    assert(opt.nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than opt.nGPU specified')
    local features_single = features
    local gpus = torch.range(1, opt.nGPU):totable()
    local dpt = nn.DataParallelTable(1, true, true)
                    :add(features_single, gpus)
                    :threads(function() 
                         if opt.backend == 'cudnn' then
                            require 'cudnn'
                            cudnn.benchmark = true 
                            cudnn.verbose = false
                         end
                     end)
    features = dpt:cuda() 
    return features
end

helpers.setMultiGPU = setMultiGPU

return helpers
