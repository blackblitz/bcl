"""Trainer."""

smi_simple = [
    'AutodiffQuadraticConsolidation', 'Finetuning',
    'ElasticWeightConsolidation', 'Joint',
    'NeuralConsolidation', 'SynapticIntelligence'
]
smi_replay = ['GDumb', 'TICReplay']
svi_simple = [
    'GMVCL', 'GVCL', 'SimpleGMSFSVI',
    'SimpleGSFSVI', 'SimpleTSFSVI', 'TVCL'
]
svi_replay = ['PriorExactGMSFSVI', 'PriorExactGSFSVI', 'PriorExactTSFSVI']

module_map = (
    dict.fromkeys(smi_simple, 'train.trainer.smi.simple')
    | dict.fromkeys(smi_replay, 'train.trainer.smi.replay')
    | dict.fromkeys(svi_simple, 'train.trainer.svi.simple')
    | dict.fromkeys(svi_replay, 'train.trainer.svi.replay')
)
