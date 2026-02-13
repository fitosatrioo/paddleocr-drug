import sys

def test_imports():
    """Test if required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = {
        'paddle': 'PaddlePaddle',
        'paddleocr': 'PaddleOCR',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - FAILED")
            failed.append(name)
    
    return len(failed) == 0, failed

def test_paddleocr():
    """Test basic PaddleOCR functionality"""
    print("\nTesting PaddleOCR initialization...")
    
    try:
        from paddleocr import PaddleOCR
        
        # Try to initialize
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False
        )
        
        print("  ✓ PaddleOCR initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ PaddleOCR initialization failed: {str(e)}")
        return False

def test_paddle_device():
    """Check Paddle device configuration"""
    print("\nChecking Paddle configuration...")
    
    try:
        import paddle
        
        print(f"  PaddlePaddle version: {paddle.__version__}")
        print(f"  Device: CPU (GPU not available/configured)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking Paddle: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("PaddleOCR Installation Test")
    print("=" * 50)
    print()
    
    # Test imports
    imports_ok, failed_packages = test_imports()
    
    # Test Paddle config
    paddle_ok = test_paddle_device()
    
    # Test PaddleOCR
    ocr_ok = test_paddleocr()
    
    # Print summary
    print("\n" + "=" * 50)
    if imports_ok and paddle_ok and ocr_ok:
        print("✓ All tests passed!")
        print("Installation is working correctly.")
        print("\nYou can now run:")
        print("  python paddle_ocr_detector.py")
    else:
        print("✗ Some tests failed!")
        if failed_packages:
            print(f"\nMissing packages: {', '.join(failed_packages)}")
        print("\nPlease run:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    print("=" * 50)

if __name__ == "__main__":
    main()
