


# CPU 
```
DS_ACCELERATOR=cpu python3.10 test_deepspeed.py --deepspeed_config=ds_config.json -p 0 --steps=200 --backend gloo
```

# XLA
```
DS_ACCELERATOR=xla python3.10 test_deepspeed.py --deepspeed_config=ds_config.json -p 0 --steps=200 --backend xla
```

# XLA SPMD
```
DS_ACCELERATOR=xla python3.10 test_spmd_deepspeed.py --deepspeed_config=ds_config.json -p 0 --steps=200
```

