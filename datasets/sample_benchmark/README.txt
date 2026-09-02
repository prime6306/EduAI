Sample Anti-Spoofing Benchmark Dataset Structure
=================================================

Directory layout:
  train/
    real/   <- Place genuine, live face crops here (160x160)
    spoof/  <- Place spoof/attack face crops here (photo prints, screens)
  val/
    real/
    spoof/

You can populate this automatically using:
  python scripts/create_dataset.py --extract-faces --input-dir <raw_photos> --output-dir datasets/sample_benchmark
Or record your own with:
  python scripts/create_dataset.py --webcam-record
