"""Training package."""

smi_simple = [
    'AutodiffQuadraticConsolidation', 'Finetuning',
    'ElasticWeightConsolidation', 'Joint',
    'NeuralConsolidation', 'SynapticIntelligence'
]
smi_replay = ['GDumb', 'TICReplay']
svi_simple = ['GMVCL', 'GVCL', 'SimpleGMSFSVI', 'SimpleGSFSVI']
svi_replay = ['PriorExactGMSFSVI', 'PriorExactGSFSVI']

module_map = (
    dict.fromkeys(smi_simple, 'train.smi.simple')
    | dict.fromkeys(smi_replay, 'train.smi.replay')
    | dict.fromkeys(svi_simple, 'train.svi.simple')
    | dict.fromkeys(svi_replay, 'train.svi.replay')
)
