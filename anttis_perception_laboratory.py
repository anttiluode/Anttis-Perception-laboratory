#!/usr/bin/env python3
"""
Antti's Perception Laboratory
A professional node-based interface for multi-domain perception experiments.

Combines: Webcam/Microphone, FFT Cochlea, 3D Scout Fields, Spectral Wiping,
Math, and Logic operators. All in one unified real-time flow.

Requirements:
  pip install PyQt6 numpy opencv-python pyqtgraph scipy Pillow pyaudio
"""

import sys
import math
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
import cv2
import pyqtgraph as pg
from scipy.fft import rfft, irfft # <-- UPDATED IMPORT
from collections import deque
from PIL import Image, ImageDraw, ImageFont # <-- ADDED IMPORT

# --- PyAudio is required for SpeakerOutputNode and Microphone Input ---
# Install it with: pip install pyaudio
try:
    import pyaudio
except ImportError:
    print("Warning: pyaudio not installed. Audio input/output nodes will be non-functional.")
    pyaudio = None
# ----------------------------------------------------------------------

pg.setConfigOptions(imageAxisOrder='row-major')

# ==================== BASE NODE SYSTEM ====================

class BaseNode:
    """Base class for all perception nodes"""
    NODE_CATEGORY = "Base"
    NODE_COLOR = QtGui.QColor(80, 80, 80)
    
    def __init__(self):
        self.inputs = {}   # {'port_name': 'port_type'}
        self.outputs = {}  # {'port_name': 'port_type'}
        self.input_data = {}
        self.node_title = "Base Node"
        
    def pre_step(self):
        """Clear input buffers before propagation"""
        self.input_data = {name: [] for name in self.inputs}
        
    def set_input(self, port_name, value, port_type='signal', coupling=1.0):
        """Receive data from connected edges"""
        if port_name not in self.input_data:
            return
        if port_type == 'signal':
            # Ensure value is treated as a single scalar for blending
            if isinstance(value, (np.ndarray, list)):
                value = value[0] if len(value) > 0 else 0.0
            self.input_data[port_name].append(float(value) * coupling)
        else:
            if value is not None:
                self.input_data[port_name].append(value)
                
    def get_blended_input(self, port_name, blend_mode='sum'):
        """Get combined input from all connections"""
        values = self.input_data.get(port_name, [])
        if not values:
            return None
            
        if blend_mode == 'sum' and isinstance(values[0], (int, float)):
            return np.sum(values)
        elif blend_mode == 'mean' and isinstance(values[0], np.ndarray):
            # Average multiple arrays
            if len(values) > 0:
                # Need to handle cases where array sizes might not match perfectly if not careful
                return np.mean([v.astype(float) for v in values if v is not None and v.size > 0], axis=0)
            return None
        return values[0]
        
    def step(self):
        """Override in subclass - main processing logic"""
        pass
        
    def get_output(self, port_name):
        """Override in subclass - return output data"""
        return None
        
    def get_display_image(self):
        """Override in subclass - return QImage for node preview"""
        return None
        
    def close(self):
        """Cleanup resources. Called when node is deleted."""
        pass

    def get_config_options(self):
        """Returns a list of (display_name, key, current_value, options) for configuration dialog."""
        return []

# ==================== SOURCE NODES ====================

class MediaSourceNode(BaseNode):
    """Source node for video (Webcam) or audio (Microphone) input."""
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80)
    
    def __init__(self, source_type='Webcam', device_id=0, width=160, height=120, sample_rate=44100):
        super().__init__()
        self.device_id = int(device_id) 
        self.source_type = source_type
        self.node_title = f"Source ({source_type})"
        self.w, self.h = width, height
        self.sample_rate = sample_rate
        
        self.outputs = {'signal': 'signal', 'image': 'image'}

        self.frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.signal_output = 0.0 
        
        self.pa = pyaudio.PyAudio() if pyaudio else None
        self.cap = None 
        self.stream = None
        
        self.setup_source()
        
    def setup_source(self):
        """Initializes or re-initializes resources based on selected type."""
        # 1. Cleanup existing resources
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.stream:
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
        
        # We don't terminate self.pa here as it's used to query devices in get_config_options
        
        self.cap = None
        self.stream = None

        try:
            if self.source_type == 'Webcam':
                self.cap = cv2.VideoCapture(self.device_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not self.cap.isOpened():
                    print(f"Warning: Cannot open webcam {self.device_id}")
            
            elif self.source_type == 'Microphone':
                if not self.pa:
                    print("Error: PyAudio not available for Microphone input.")
                    return
                
                channels = 1
                
                self.stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=channels, 
                    rate=int(self.sample_rate),
                    input=True,
                    input_device_index=self.device_id,
                    frames_per_buffer=1024
                )
        except Exception as e:
            print(f"Error setting up source {self.source_type}: {e}")
            self.node_title = f"Source ({self.source_type} ERROR)"
            return
            
        self.node_title = f"Source ({self.source_type})"

    def step(self):
        self.frame *= 0 # clear frame to black
        
        if self.source_type == 'Webcam' and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.resize(frame, (self.w, self.h))
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.signal_output = np.mean(gray) / 255.0 # Luminance signal
                
        elif self.source_type == 'Microphone' and self.stream and self.stream.is_active():
            try:
                data = self.stream.read(256, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if audio_data.size > 0:
                    self.signal_output = np.sqrt(np.mean(audio_data**2)) * 5.0 
                
                # Visual Feedback
                if audio_data.size > 0:
                    padded_audio = np.pad(audio_data, (0, 1024 - len(audio_data)))
                    spec = np.abs(np.fft.fft(padded_audio))
                    spec = spec[:self.w].copy() 
                    
                    spec = np.log1p(spec)
                    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
                    
                    audio_img = np.zeros((self.h, self.w), dtype=np.uint8)
                    for i in range(self.w):
                        h = int(spec[i] * self.h)
                        audio_img[self.h - h:, i] = 255
                    
                    self.frame = cv2.cvtColor(audio_img, cv2.COLOR_GRAY2BGR)
                    
            except Exception:
                self.signal_output = 0.0
        
    def get_output(self, port_name):
        if port_name == 'image':
            if self.frame.ndim == 3:
                # Convert BGR/RGB frame to normalized grayscale image (0-1 float)
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            else:
                gray = self.frame.astype(np.float32) / 255.0
            return gray
        elif port_name == 'signal':
            return self.signal_output
        return None
        
    def get_display_image(self):
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        
    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.stream:
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
        if self.pa:
            try: self.pa.terminate() # Terminate the local PyAudio instance
            except Exception: pass
        super().close()
        
    def get_config_options(self):
        webcam_devices = [("Default Webcam (0)", 0), ("Secondary Webcam (1)", 1)]
        mic_devices = []
        if self.pa:
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    mic_devices.append((f"{info['name']} ({i})", i))
        
        device_options = mic_devices if self.source_type == 'Microphone' else webcam_devices
        
        if not any(v == self.device_id for _, v in device_options):
             device_options.append((f"Selected Device ({self.device_id})", self.device_id))
        
        return [
            ("Source Type", "source_type", self.source_type, [("Webcam", "Webcam"), ("Microphone", "Microphone")]),
            ("Device ID", "device_id", self.device_id, device_options),
        ]

class NoiseGeneratorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(40, 120, 80)
    
    def __init__(self, width=160, height=120, noise_type='white', speed=0.1):
        super().__init__()
        self.node_title = "Noise Gen"
        self.outputs = {'image': 'image', 'signal': 'signal'} 
        self.w, self.h = width, height
        self.noise_type = noise_type 
        self.speed = speed
        
        self.img = np.random.rand(self.h, self.w).astype(np.float32)
        self.signal_value = 0.0 
        
        self.brown_state = np.zeros((self.h, self.w), dtype=np.float32)
        self.perlin_phase = np.random.rand(2) * 100

    def _generate_noise_step(self, shape):
        """Generates a noise array based on the selected type."""
        if self.noise_type == 'white':
            return np.random.rand(*shape)
        
        elif self.noise_type == 'brown':
            rand_step = np.random.randn(*shape) * 0.05 * self.speed
            self.brown_state = self.brown_state + rand_step
            self.brown_state = np.clip(self.brown_state, -1.0, 1.0)
            return (self.brown_state + 1.0) / 2.0
        
        elif self.noise_type == 'perlin':
            X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            self.perlin_phase += self.speed * 0.1 
            
            noise_val = (
                np.sin(X * 0.1 + self.perlin_phase[0]) + 
                np.sin(Y * 0.05 + self.perlin_phase[1] * 0.5)
            )
            noise_val = (noise_val - noise_val.min()) / (noise_val.max() - noise_val.min() + 1e-9)
            noise_val += np.random.rand(*shape) * 0.01 
            return np.clip(noise_val, 0, 1)
            
        elif self.noise_type == 'quantum':
            noise = np.random.rand(*shape)
            if np.random.rand() < 0.02 * self.speed * 10: 
                 noise += np.random.rand(*shape) * 0.5 * self.speed
            return np.clip(noise, 0, 1)
            
        return np.random.rand(*shape)

    def step(self):
        new_noise = self._generate_noise_step((self.h, self.w))
        
        self.img = self.img * (1.0 - self.speed) + new_noise * self.speed
        
        center_y, center_x = self.h // 2, self.w // 2
        window_size = 10
        center_patch = self.img[
            center_y - window_size//2 : center_y + window_size//2,
            center_x - window_size//2 : center_x + window_size//2
        ]
        self.signal_value = np.mean(center_patch) * 2.0 - 1.0
        
    def get_output(self, port_name):
        if port_name == 'image':
            return self.img
        elif port_name == 'signal':
            return self.signal_value
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Noise Type", "noise_type", self.noise_type, [
                ("White (Uniform)", "white"), 
                ("Brown (Coherent)", "brown"),
                ("Perlin (Pattern)", "perlin"), 
                ("Quantum (Spikes)", "quantum")
            ]),
            ("Speed (Blend Factor)", "speed", self.speed, None)
        ]

# ==================== TRANSFORM NODES ====================

class FFTCochleaNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40)
    
    def __init__(self, freq_bins=64):
        super().__init__()
        self.node_title = "FFT Cochlea"
        self.inputs = {'image': 'image', 'signal': 'signal'}
        # ADDED 'complex_spectrum' OUTPUT
        self.outputs = {
            'spectrum': 'spectrum', 
            'signal': 'signal', 
            'image': 'image', 
            'complex_spectrum': 'complex_spectrum' # <-- ADDED
        }
        
        self.freq_bins = freq_bins
        self.buffer = np.zeros(128, dtype=np.float32)
        self.x = 0.0
        self.internal_freq = np.random.uniform(2.0, 15.0)
        # 64x64 output image size for the spectral map
        self.cochlea_img = np.zeros((64, 64), dtype=np.uint8) 
        self.spectrum_data = None
        self.complex_spectrum_data = None # <-- ADDED
        
    def step(self):
        u = self.get_blended_input('signal', 'sum') or 0.0
        
        alpha = 0.45
        decay = 0.92
        gain = 0.9
        
        newx = decay * self.x + gain * math.tanh(u + alpha * self.x)
        self.x = newx
        
        self.buffer *= 0.998
        if abs(self.x) > 0.09:
            amp = np.tanh(self.x) * 0.25
            t = np.linspace(0, 1, 10)
            sig = amp * np.sin(2*np.pi*(self.internal_freq + amp*10) * t)
            self.buffer[:-len(sig)] = self.buffer[len(sig):]
            self.buffer[-len(sig):] = sig
            
        img = self.get_blended_input('image', 'mean')
        if img is not None:
            self.compute_image_spectrum(img)
        else:
            self.compute_buffer_spectrum()
            
    def compute_buffer_spectrum(self):
        # Fallback for signal input: standard FFT
        f = np.fft.fft(self.buffer)
        fsh = np.fft.fftshift(f)
        mag = np.abs(fsh)
        center = len(mag)//2
        half = min(self.freq_bins//2, center-1)
        spec = mag[center-half:center+half]
        self.spectrum_data = spec
        self.complex_spectrum_data = None # <-- ADDED
        self.update_display_from_spectrum(spec)
        
    def compute_image_spectrum(self, img):
        # Primary image input: row-wise FFT (Cochlea map)
        if img.ndim != 2:
            return
        
        # Calculate FFT across image rows
        spec = rfft(img.astype(np.float64), axis=1)
        self.complex_spectrum_data = spec.copy() # <-- ADDED: Store raw complex data
        mag = np.abs(spec)
        
        # Downsample/select frequencies if needed
        if mag.shape[1] > self.freq_bins:
            indices = np.linspace(0, mag.shape[1]-1, self.freq_bins).astype(int)
            mag = mag[:, indices]
        
        self.spectrum_data = np.mean(mag, axis=0) # Average spectrum for 'spectrum' output
        
        # Prepare for image output/display
        display = np.log1p(mag)
        display = (display - display.min()) / (display.max() - display.min() + 1e-9)
        
        # Resize the display map to the internal display size (64x64)
        h_target, w_target = self.cochlea_img.shape
        display_u8 = (display * 255).astype(np.uint8)
        self.cochlea_img = cv2.resize(display_u8, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        
    def update_display_from_spectrum(self, spec):
        # Update cochlea_img for the display when only signal is available
        arr = np.log1p(spec)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        
        w, h = self.cochlea_img.shape
        self.cochlea_img = np.zeros((h, w), dtype=np.uint8)
        
        # Draw the spectrum as vertical bars
        for i in range(min(len(arr), w)):
            v = int(255 * arr[i])
            self.cochlea_img[h - v:, i] = 255
        self.cochlea_img = np.flipud(self.cochlea_img)
        
    def get_output(self, port_name):
        if port_name == 'spectrum':
            return self.spectrum_data
        elif port_name == 'signal':
            return self.x
        elif port_name == 'image': # NEW: Normalized spectral map image
            return self.cochlea_img.astype(np.float32) / 255.0
        elif port_name == 'complex_spectrum': # <-- ADDED
            return self.complex_spectrum_data
        return None
        
    def get_display_image(self):
        img = np.ascontiguousarray(self.cochlea_img)
        h, w = img.shape
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        
    def randomize(self):
        self.internal_freq = np.random.uniform(2.0, 15.0)
        self.x = np.random.uniform(-0.5, 0.5)

class iFFTCochleaNode(BaseNode):
    """
    Performs an Inverse Real FFT on a complex spectrum (from FFTCochleaNode)
    to reconstruct a 2D image.
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(200, 100, 60)
    
    def __init__(self, height=120, width=160):
        super().__init__()
        self.node_title = "iFFT Cochlea"
        self.inputs = {'complex_spectrum': 'complex_spectrum'}
        self.outputs = {'image': 'image'}
        
        self.h, self.w = height, width
        self.reconstructed_img = np.zeros((self.h, self.w), dtype=np.float32)

    def step(self):
        complex_spec = self.get_blended_input('complex_spectrum', 'mean')
        
        if complex_spec is not None and complex_spec.ndim == 2:
            try:
                # Perform inverse real FFT
                img = irfft(complex_spec, axis=1).astype(np.float32)
                
                # Resize to target output size (just in case)
                self.reconstructed_img = cv2.resize(img, (self.w, self.h))
                
                # Normalize for viewing (0-1)
                min_v, max_v = np.min(self.reconstructed_img), np.max(self.reconstructed_img)
                if (max_v - min_v) > 1e-6:
                    self.reconstructed_img = (self.reconstructed_img - min_v) / (max_v - min_v)
                else:
                    self.reconstructed_img.fill(0.5)
                    
            except Exception as e:
                print(f"iFFT Error: {e}")
                self.reconstructed_img.fill(0.0)
        else:
            # Fade to black if no input
            self.reconstructed_img *= 0.9 
            
    def get_output(self, port_name):
        if port_name == 'image':
            return self.reconstructed_img
        return None
        
    def get_display_image(self):
        img_u8 = (np.clip(self.reconstructed_img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Height", "height", self.h, None),
            ("Width", "width", self.w, None)
        ]

class SpectralWipeNode(BaseNode):
    """
    Applies a moving window (bandpass, lowpass, highpass) to an input image.
    Can be controlled by a signal or sweeps automatically.
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(60, 180, 160)
    
    def __init__(self, wipe_width=20, mode='bandpass'):
        super().__init__()
        self.node_title = "Spectral Wipe"
        self.inputs = {'image': 'image', 'wipe_pos': 'signal'}
        self.outputs = {'image': 'image', 'signal': 'signal'}
        
        self.wipe_width = int(wipe_width)
        self.mode = mode
        self.wipe_pos = 0.5 # Internal 0-1 position
        self.img_out = np.zeros((64, 64), dtype=np.float32)
        self.signal_out = 0.0

    def step(self):
        img = self.get_blended_input('image', 'mean')
        pos_in = self.get_blended_input('wipe_pos', 'sum')
        
        # Update wipe position
        if pos_in is not None:
            # Map input signal (-1 to 1) to 0-1
            self.wipe_pos = np.clip((pos_in + 1.0) / 2.0, 0.0, 1.0)
        else:
            # No input, use internal LFO
            self.wipe_pos = (self.wipe_pos + 0.005) % 1.0
            
        if img is not None:
            if img.shape != self.img_out.shape:
                self.img_out = np.zeros(img.shape, dtype=np.float32)

            h, w = img.shape
            mask = np.zeros_like(img)
            
            center_pix = int(self.wipe_pos * w)
            half_width = self.wipe_width // 2
            
            if self.mode == 'bandpass':
                start = max(0, center_pix - half_width)
                end = min(w, center_pix + half_width)
                mask[:, start:end] = 1.0
            elif self.mode == 'lowpass':
                mask[:, :center_pix] = 1.0
            elif self.mode == 'highpass':
                mask[:, center_pix:] = 1.0
            
            self.img_out = img * mask
            self.signal_out = np.mean(self.img_out)
        else:
            self.img_out *= 0.9
            self.signal_out *= 0.9
            
    def get_output(self, port_name):
        if port_name == 'image':
            return self.img_out
        elif port_name == 'signal':
            return self.signal_out
        return None
        
    def get_display_image(self):
        h, w = self.img_out.shape
        img_u8 = (np.clip(self.img_out, 0, 1) * 255).astype(np.uint8)
        
        # Add a visual marker for the wipe position
        marker_x = int(self.wipe_pos * w)
        marker_x = np.clip(marker_x, 0, w-1)
        img_u8[0:3, marker_x] = 255
        img_u8[h-3:h, marker_x] = 255
        
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Wipe Width", "wipe_width", self.wipe_width, None),
            ("Mode", "mode", self.mode, [
                ("Band-Pass", "bandpass"),
                ("Low-Pass", "lowpass"),
                ("High-Pass", "highpass")
            ])
        ]

class SignalProcessorNode(BaseNode):
    """Takes a signal and processes it through different algorithms."""
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 180, 250)
    
    def __init__(self, processing_mode='smoothing', factor=0.1):
        super().__init__()
        self.node_title = "Signal Processor"
        self.inputs = {'input_signal': 'signal'}
        self.outputs = {'output_signal': 'signal'}
        
        self.processing_mode = processing_mode
        self.factor = float(factor) # <-- MODIFIED: Ensure float
        self.last_input = 0.0
        self.integrated_state = 0.0
        self.processed_output = 0.0
        
    def step(self):
        u = self.get_blended_input('input_signal', 'sum') or 0.0
        
        output = u
        
        if self.processing_mode == 'smoothing':
            alpha = np.clip(self.factor, 0.0, 1.0) # Smoothing factor
            self.processed_output = self.processed_output * (1.0 - alpha) + u * alpha
            output = self.processed_output
            
        elif self.processing_mode == 'differentiation':
            # Factor acts as sensitivity (1/dt)
            output = (u - self.last_input) * (1.0 / max(self.factor, 1e-6)) 
            self.processed_output = output
            
        elif self.processing_mode == 'integration':
            # Factor acts as decay speed
            decay = np.clip(1.0 - self.factor * 0.1, 0.9, 1.0) 
            self.integrated_state = self.integrated_state * decay + u * 0.05
            output = self.integrated_state
            self.processed_output = output
            
        elif self.processing_mode == 'high_pass': # <-- ADDED
            # 1st order IIR high-pass. Factor is (1-alpha)
            alpha = np.clip(1.0 - self.factor, 0.01, 0.99)
            self.processed_output = alpha * (self.processed_output + u - self.last_input)
            output = self.processed_output

        elif self.processing_mode == 'full_wave_rectify': # <-- ADDED
            # Factor is unused
            output = np.abs(u)
            self.processed_output = output

        elif self.processing_mode == 'tanh_distortion': # <-- ADDED
            # Factor acts as gain/drive
            gain = max(self.factor, 1e-6)
            output = np.tanh(u * gain)
            self.processed_output = output

        self.last_input = u
        
    def get_output(self, port_name):
        if port_name == 'output_signal':
            return self.processed_output
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Simple bar display of the processed output
        v = np.clip(self.processed_output, -1.0, 1.0)
        bar_height = int((v + 1.0) / 2.0 * h)
        
        img[h - bar_height:, w//2 - 2 : w//2 + 2] = 255
        img[h//2 - 1 : h//2 + 1, :] = 80 # Center line

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Mode", "processing_mode", self.processing_mode, [
                ("Smoothing (EMA)", "smoothing"), 
                ("Differentiation", "differentiation"),
                ("Integration (Decay)", "integration"),
                ("High-Pass Filter", "high_pass"),       # <-- ADDED
                ("Full Wave Rectify", "full_wave_rectify"), # <-- ADDED
                ("Tanh Distortion", "tanh_distortion")   # <-- ADDED
            ]),
            # Factor interpretation changes based on mode
            ("Factor", "factor", self.factor, None)
        ]

class SignalMathNode(BaseNode):
    """Performs a mathematical operation on two input signals."""
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 180, 250)
    
    def __init__(self, operation='add'):
        super().__init__()
        self.node_title = "Signal Math"
        self.inputs = {'A': 'signal', 'B': 'signal'}
        self.outputs = {'result': 'signal'}
        
        self.operation = operation
        self.result = 0.0
        self.last_a = 0.0
        self.last_b = 0.0

    def step(self):
        # Use last known value if an input is disconnected
        a = self.get_blended_input('A', 'sum')
        b = self.get_blended_input('B', 'sum')
        
        if a is None: a = self.last_a
        else: self.last_a = a
        
        if b is None: b = self.last_b
        else: self.last_b = b
        
        if self.operation == 'add':
            self.result = a + b
        elif self.operation == 'subtract':
            self.result = a - b
        elif self.operation == 'multiply':
            self.result = a * b
        elif self.operation == 'divide':
            if abs(b) < 1e-6:
                self.result = 0.0
            else:
                self.result = a / b
        
    def get_output(self, port_name):
        if port_name == 'result':
            return self.result
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        op_symbol = {
            'add': '+', 'subtract': '-', 'multiply': '×', 'divide': '÷'
        }.get(self.operation, '?')
        
        text = f"A {op_symbol} B\n= {self.result:.2f}"
        
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            # Use a basic font if default is unavailable
            font = ImageFont.load_default()
        except IOError:
            font = None 
            
        draw.text((5, 20), text, fill=255, font=font)
        
        # Convert back to numpy array for QImage
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def get_config_options(self):
        return [
            ("Operation", "operation", self.operation, [
                ("Add (A + B)", "add"),
                ("Subtract (A - B)", "subtract"),
                ("Multiply (A × B)", "multiply"),
                ("Divide (A ÷ B)", "divide")
            ])
        ]

class SignalLogicNode(BaseNode):
    """
    Outputs one of two signals based on a test condition.
    (If Test > Threshold, output A, else output B)
    """
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(100, 180, 250)
    
    def __init__(self, threshold=0.5, condition='>'):
        super().__init__()
        self.node_title = "Signal Logic"
        self.inputs = {'test': 'signal', 'if_true': 'signal', 'if_false': 'signal'}
        self.outputs = {'result': 'signal'}
        
        self.threshold = float(threshold)
        self.condition = condition
        self.result = 0.0
        self.last_true = 0.0
        self.last_false = 0.0
        self.condition_met = False

    def step(self):
        test_val = self.get_blended_input('test', 'sum') or 0.0
        if_true_val = self.get_blended_input('if_true', 'sum')
        if_false_val = self.get_blended_input('if_false', 'sum')
        
        # Hold last valid values
        if if_true_val is not None: self.last_true = if_true_val
        if if_false_val is not None: self.last_false = if_false_val
        
        self.condition_met = False
        if self.condition == '>':
            self.condition_met = test_val > self.threshold
        elif self.condition == '<':
            self.condition_met = test_val < self.threshold
        elif self.condition == '==':
            self.condition_met = abs(test_val - self.threshold) < 1e-6
            
        self.result = self.last_true if self.condition_met else self.last_false

    def get_output(self, port_name):
        if port_name == 'result':
            return self.result
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Green light for TRUE, Red for FALSE
        if self.condition_met:
            img[10:h-10, 10:w-10] = (60, 220, 60) # Green
            text = "TRUE"
        else:
            img[10:h-10, 10:w-10] = (220, 60, 60) # Red
            text = "FALSE"
            
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except IOError:
            font = None
            
        condition_text = f"Test {self.condition} {self.threshold}"
        draw.text((5, 2), condition_text, fill=(255,255,255), font=font)
        draw.text((18, 28), text, fill=(255,255,255), font=font)
        
        img = np.array(img_pil)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Condition", "condition", self.condition, [
                ("Greater Than (>) ", ">"),
                ("Less Than (<)", "<"),
                ("Equals (==)", "==")
            ]),
            ("Threshold", "threshold", self.threshold, None)
        ]

class Scout3DNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(220, 120, 40)
    
    def __init__(self, scout_count=500):
        super().__init__()
        self.node_title = "3D Scouts"
        self.inputs = {'image': 'image', 'spectrum': 'spectrum'}
        self.outputs = {'density': 'signal'}
        
        self.scout_count = scout_count
        self.positions = np.random.randn(scout_count, 3).astype(np.float32) * 20
        self.velocities = np.zeros((scout_count, 3), dtype=np.float32)
        self.colors = np.random.rand(scout_count, 3).astype(np.float32)
        self.density = 0.0
        
    def step(self):
        img = self.get_blended_input('image', 'mean')
        spec = self.get_blended_input('spectrum', 'mean')
        
        if img is not None:
            strength = np.mean(img)
            center = np.array([0, 0, 0])
            for i in range(self.scout_count):
                to_center = center - self.positions[i]
                self.velocities[i] += to_center * strength * 0.01
                
        if spec is not None and spec.ndim == 1:
            mean_spec = np.mean(spec)
            self.positions[:, 2] += (mean_spec * 0.1 - 0.05)
            
        self.velocities *= 0.95
        self.positions += self.velocities
        
        self.positions = np.clip(self.positions, -50, 50)
        
        self.density = np.mean(np.linalg.norm(self.velocities, axis=1))
        
    def get_output(self, port_name):
        if port_name == 'density':
            return self.density
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        for pos in self.positions:
            x = int((pos[0] + 50) / 100 * w)
            y = int((pos[1] + 50) / 100 * h)
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = min(255, img[y, x] + 50)
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

class Attractor3DNode(BaseNode):
    """3D particle dynamics driven by two signal inputs (like slider 2)"""
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(180, 80, 180) 
    
    def __init__(self, particle_count=300):
        super().__init__()
        self.node_title = "3D Attractor"
        self.inputs = {'input_1': 'signal', 'input_2': 'signal'} 
        self.outputs = {'signal': 'signal'}
        
        self.particle_count = particle_count
        self.positions = np.random.randn(particle_count, 3).astype(np.float32) * 5
        self.velocities = np.zeros((particle_count, 3), dtype=np.float32)
        self.output_signal = 0.0
        self.zoom_factor = 1.0 
        
    def step(self):
        in1 = self.get_blended_input('input_1', 'sum') or 0.0
        in2 = self.get_blended_input('input_2', 'sum') or 0.0
        
        attraction = np.tanh(in1 * 0.5) * 0.05
        
        center = np.mean(self.positions, axis=0)
        
        for i in range(self.particle_count):
            pos = self.positions[i]
            
            to_center = center - pos
            self.velocities[i] += to_center * attraction
            
            self.velocities[i] -= pos * 0.01
            
            self.velocities[i, 2] += np.tanh(in2 * 0.1) * 0.1
        
        self.velocities *= 0.9
        self.positions += self.velocities
        
        self.output_signal = np.std(self.positions)
        
    def get_output(self, port_name):
        if port_name == 'signal':
            return self.output_signal
        return None
        
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        visual_range = 10.0 * self.zoom_factor 
        
        for pos in self.positions:
            x = int((pos[0] + visual_range) / (visual_range * 2) * w)
            y = int((pos[1] + visual_range) / (visual_range * 2) * h)
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = min(255, img[y, x] + 20)
                
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

# ==================== OUTPUT NODES ====================

class ImageDisplayNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120)
    
    def __init__(self, width=160, height=120):
        super().__init__()
        self.node_title = "Display"
        self.inputs = {'image': 'image'}
        self.w, self.h = width, height
        self.img = np.zeros((self.h, self.w), dtype=np.float32)
        
    def step(self):
        img = self.get_blended_input('image', 'mean')
        if img is not None:
            if img.shape != (self.h, self.w):
                img = cv2.resize(img, (self.w, self.h))
            self.img = img
        else:
            self.img *= 0.95
            
    def get_display_image(self):
        img_u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(img_u8)
        return QtGui.QImage(img_u8.data, self.w, self.h, self.w, QtGui.QImage.Format.Format_Grayscale8)

class SignalMonitorNode(BaseNode):
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120)
    
    def __init__(self, history_len=500):
        super().__init__()
        self.node_title = "Signal Monitor"
        self.inputs = {'signal': 'signal'}
        self.history = deque(maxlen=history_len)
        self.history_len = history_len
        self.plot_widget = None 
        self.plot_curve = None 
        
    def step(self):
        val = self.get_blended_input('signal', 'sum') or 0.0
        self.history.append(val)
            
    def get_display_image(self):
        w, h = 64, 32
        img = np.zeros((h, w), dtype=np.uint8)
        if len(self.history) > 1:
            history_array = np.array(list(self.history))
            
            min_val = np.min(history_array)
            max_val = np.max(history_array)
            range_val = max_val - min_val
            
            if range_val > 1e-6:
                vis_history = (history_array - min_val) / range_val
            else:
                vis_history = np.full_like(history_array, 0.5) 
            
            for i in range(min(len(vis_history) - 1, w - 1)):
                val1 = vis_history[-(i+1)]
                
                y1 = int((1 - val1) * (h-1)) 
                
                x1 = w - 1 - i
                
                y1 = max(0, min(h-1, y1))
                img[y1, x1] = 255

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)

    def close(self):
        if self.plot_widget:
            self.plot_widget.close()
        self.plot_curve = None
        super().close()


class SpeakerOutputNode(BaseNode):
    """Real signal output device using PyAudio."""
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(120, 40, 120) 
    
    def __init__(self, buffer_len=512, sample_rate=44100, device_index=None):
        super().__init__()
        self.node_title = "Speaker Output"
        self.inputs = {'signal': 'signal'}
        self.buffer = deque(maxlen=buffer_len)
        self.rms_level = 0.0
        
        self.pa = pyaudio.PyAudio() if pyaudio else None
        self.sample_rate = int(sample_rate)
        self.stream = None
        self.device_index = device_index
        self.is_playing = False
        
        if self.pa and self.device_index is None:
            try:
                self.device_index = self.pa.get_default_output_device_info()['index']
            except Exception:
                self.device_index = -1 
        self.open_stream()
        
    def open_stream(self):
        """Opens or re-opens the PyAudio stream."""
        if self.stream: 
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
            
        self.is_playing = False
        if not self.pa or self.device_index < 0:
            self.node_title = "Speaker (NO PA)"
            return
            
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_index,
                frames_per_buffer=64 
            )
            self.is_playing = True
            try:
                device_name = self.pa.get_device_info_by_index(self.device_index)['name']
                self.node_title = f"Speaker ({device_name[:15]}...)"
            except:
                self.node_title = "Speaker (Active)"
            
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.stream = None
            self.node_title = "Speaker (ERROR)"
            
    def step(self):
        val = self.get_blended_input('signal', 'sum') or 0.0
        
        clipped_val = np.clip(val, -1.0, 1.0)
        self.buffer.append(clipped_val)
        
        if len(self.buffer) > 0:
            rms_val = np.sqrt(np.mean(np.square(np.array(list(self.buffer)))))
            self.rms_level = self.rms_level * 0.8 + rms_val * 0.2
        
        if self.stream and self.is_playing and len(self.buffer) >= 64:
            audio_data = np.array(list(self.buffer)[:64], dtype=np.float32)
            audio_data = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
            
            try:
                self.stream.write(audio_data.tobytes())
                for _ in range(64):
                    self.buffer.popleft() 
            except IOError:
                self.is_playing = False 
                
    def get_display_image(self):
        w, h = 64, 64
        img = np.zeros((h, w), dtype=np.uint8)
        
        buffer_array = np.array(list(self.buffer))
        if len(buffer_array) > w:
            vis_data = buffer_array[-w:]
            vis_data = (vis_data + 1.0) / 2.0 * (h-1)
            for i in range(w):
                y = int(h - 1 - vis_data[i]) 
                y = max(0, min(h-1, y))
                img[y, i] = 255
                
        bar_height = int(np.clip(self.rms_level * 5.0, 0.0, 1.0) * h)
        img[h - bar_height:, w-5:w-1] = 180 

        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        
    def get_config_options(self):
        if not self.pa:
            return [("PyAudio Not Found", "error", "Install PyAudio", [])]
            
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['max_output_channels'] > 0:
                devices.append((f"{info['name']} ({i})", i))
            
        if not any(v == self.device_index for _, v in devices):
            devices.append((f"Selected Device ({self.device_index})", self.device_index))
            
        return [
            ("Output Device", "device_index", self.device_index, devices),
            ("Sample Rate", "sample_rate", self.sample_rate, None)
        ]
        
    def close(self):
        # FIX: Ensure all PyAudio resources are released
        if self.stream:
            try: self.stream.stop_stream(); self.stream.close()
            except Exception: pass
        if self.pa:
            try: self.pa.terminate() # Terminate the local PyAudio instance
            except Exception: pass
        super().close()

# ==================== NODE REGISTRY ====================

NODE_TYPES = {
    'Media Source': MediaSourceNode, 
    'Noise Generator': NoiseGeneratorNode,
    'FFT Cochlea': FFTCochleaNode,
    'iFFT Cochlea': iFFTCochleaNode,       # <-- ADDED
    'Spectral Wipe': SpectralWipeNode,     # <-- ADDED
    'Signal Processor': SignalProcessorNode,
    'Signal Math': SignalMathNode,         # <-- ADDED
    'Signal Logic': SignalLogicNode,       # <-- ADDED
    '3D Scouts': Scout3DNode,
    '3D Attractor': Attractor3DNode, 
    'Image Display': ImageDisplayNode,
    'Signal Monitor': SignalMonitorNode,
    'Speaker Output': SpeakerOutputNode, 
}

PORT_COLORS = {
    'signal': QtGui.QColor(200, 200, 200),
    'image': QtGui.QColor(100, 150, 255),
    'spectrum': QtGui.QColor(255, 150, 100),
    'complex_spectrum': QtGui.QColor(255, 100, 255), # <-- ADDED
}

# ==================== GRAPHICS ITEMS & DIALOGS (updated) ====================

PORT_RADIUS = 7
NODE_W, NODE_H = 200, 160

class PortItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent, name, port_type, is_output=False):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, PORT_RADIUS*2, PORT_RADIUS*2, parent)
        self.name = name
        self.port_type = port_type
        self.is_output = is_output
        self.base_color = PORT_COLORS.get(port_type, QtGui.QColor(255, 0, 0))
        self.setBrush(QtGui.QBrush(self.base_color))
        self.setZValue(3)
        self.setAcceptHoverEvents(True)
        
    def hoverEnterEvent(self, ev):
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 60)))
    def hoverLeaveEvent(self, ev):
        self.setBrush(QtGui.QBrush(self.base_color))

class EdgeItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, src_port, tgt_port=None):
        super().__init__()
        self.src = src_port
        self.tgt = tgt_port
        self.port_type = src_port.port_type
        self.setZValue(1)
        self.effect_val = 0.0
        pen = QtGui.QPen(PORT_COLORS.get(self.port_type, QtGui.QColor(200,200,200)))
        pen.setWidthF(2.0)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        
    def update_path(self):
        sp = self.src.scenePos()
        tp = self.tgt.scenePos() if self.tgt else sp
        path = QtGui.QPainterPath()
        path.moveTo(sp)
        dx = (tp.x() - sp.x()) * 0.5
        c1 = QtCore.QPointF(sp.x() + dx, sp.y())
        c2 = QtCore.QPointF(tp.x() - dx, tp.y())
        path.cubicTo(c1, c2, tp)
        self.setPath(path)
        self.update_style()
        
    def update_style(self):
        val = np.clip(self.effect_val, 0.0, 1.0)
        alpha = int(80 + val * 175)
        w = 2.0 + val * 4.0
        color = PORT_COLORS.get(self.port_type, QtGui.QColor(200,200,200)).lighter(130)
        color.setAlpha(alpha)
        pen = QtGui.QPen(color)
        pen.setWidthF(w)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

class NodeItem(QtWidgets.QGraphicsItem):
    def __init__(self, sim_node):
        super().__init__()
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.sim = sim_node
        self.in_ports = {}
        self.out_ports = {}
        
        y_in = 40
        for name, ptype in self.sim.inputs.items():
            port = PortItem(self, name, ptype, is_output=False)
            port.setPos(0, y_in)
            self.in_ports[name] = port
            y_in += 25
            
        y_out = 40
        for name, ptype in self.sim.outputs.items():
            port = PortItem(self, name, ptype, is_output=True)
            port.setPos(NODE_W, y_out)
            self.out_ports[name] = port
            y_out += 25
            
        self.rect = QtCore.QRectF(0, 0, NODE_W, NODE_H)
        self.setZValue(2)
        self.display_pix = None
        
        self.random_btn_rect = None
        self.zoom_in_rect = None 
        self.zoom_out_rect = None 
        
        if isinstance(self.sim, FFTCochleaNode):
            self.random_btn_rect = QtCore.QRectF(NODE_W - 18, 4, 14, 14)
        elif isinstance(self.sim, Attractor3DNode): 
            self.zoom_in_rect = QtCore.QRectF(NODE_W - 38, 4, 14, 14) 
            self.zoom_out_rect = QtCore.QRectF(NODE_W - 18, 4, 14, 14) 
            
    def mousePressEvent(self, ev):
        if self.random_btn_rect and self.random_btn_rect.contains(ev.pos()):
            if hasattr(self.sim, 'randomize'):
                self.sim.randomize()
            ev.accept()
            return
        
        if self.zoom_in_rect and self.zoom_in_rect.contains(ev.pos()):
            self.sim.zoom_factor = max(0.1, self.sim.zoom_factor / 1.2) 
            self.update_display()
            ev.accept()
            return
        if self.zoom_out_rect and self.zoom_out_rect.contains(ev.pos()):
            self.sim.zoom_factor = min(5.0, self.sim.zoom_factor * 1.2) 
            self.update_display()
            ev.accept()
            return
            
        super().mousePressEvent(ev)
        
    def boundingRect(self):
        return self.rect.adjusted(-8, -8, 8, 8)
        
    def paint(self, painter, option, widget):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        base = self.sim.NODE_COLOR
        if self.isSelected():
            base = base.lighter(150)
        painter.setBrush(QtGui.QBrush(base))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 2))
        painter.drawRoundedRect(self.rect, 10, 10)
        
        painter.setPen(QtGui.QColor(240, 240, 240))
        font = QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QtCore.QRectF(8, 4, NODE_W-24, 20), self.sim.node_title)
        
        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.setFont(QtGui.QFont("Arial", 7))
        painter.drawText(QtCore.QRectF(8, 18, NODE_W-16, 12), self.sim.NODE_CATEGORY)
        
        painter.setFont(QtGui.QFont("Arial", 7))
        for name, port in self.in_ports.items():
            painter.drawText(port.pos() + QtCore.QPointF(12, 4), name)
        for name, port in self.out_ports.items():
            w = painter.fontMetrics().boundingRect(name).width()
            painter.drawText(port.pos() + QtCore.QPointF(-w - 12, 4), name)
            
        if self.display_pix:
            img_h = NODE_H - 50
            img_w = NODE_W - 16
            target = QtCore.QRectF(8, 38, img_w, img_h)
            scaled = self.display_pix.scaled(
                int(img_w), int(img_h),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation
            )
            x = 8 + (img_w - scaled.width()) / 2
            y = 38 + (img_h - scaled.height()) / 2
            painter.drawPixmap(QtCore.QRectF(x, y, scaled.width(), scaled.height()),
                               scaled, QtCore.QRectF(scaled.rect()))
                                
        if self.random_btn_rect:
            painter.setBrush(QtGui.QColor(255, 200, 60))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.random_btn_rect)
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.random_btn_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "R")
            
        if self.zoom_in_rect and self.zoom_out_rect: 
            # Draw Zoom In button (-)
            painter.setBrush(QtGui.QColor(60, 180, 255))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.zoom_in_rect)
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.zoom_in_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "-")

            # Draw Zoom Out button (+)
            painter.setBrush(QtGui.QColor(60, 180, 255))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.zoom_out_rect)
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.zoom_out_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "+")
                                
    def update_display(self):
        qimg = self.sim.get_display_image()
        if qimg:
            self.display_pix = QtGui.QPixmap.fromImage(qimg)
        self.update()

class NodeConfigDialog(QtWidgets.QDialog):
    def __init__(self, node_item, parent=None):
        super().__init__(parent)
        self.node = node_item.sim
        self.setWindowTitle(f"Configure: {self.node.node_title}")
        self.setFixedWidth(300)
        
        layout = QtWidgets.QVBoxLayout(self)
        self.inputs = {}

        for display_name, key, current_value, options in self.node.get_config_options():
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(QtWidgets.QLabel(display_name + ":"))

            if options:
                combo = QtWidgets.QComboBox()
                for name, value in options:
                    combo.addItem(name, userData=value)
                    if value == current_value:
                        combo.setCurrentIndex(combo.count() - 1)
                
                if isinstance(current_value, (int, float)) and not any(v == current_value for _, v in options):
                    combo.addItem(f"Selected Device ({current_value})", userData=current_value)
                    combo.setCurrentIndex(combo.count() - 1)
                    
                h_layout.addWidget(combo, 1)
                self.inputs[key] = combo
            else:
                line_edit = QtWidgets.QLineEdit(str(current_value))
                h_layout.addWidget(line_edit, 1)
                self.inputs[key] = line_edit
                
            layout.addLayout(h_layout)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_new_config(self):
        new_config = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, QtWidgets.QComboBox):
                new_config[key] = widget.currentData()
            elif isinstance(widget, QtWidgets.QLineEdit):
                text = widget.text()
                try:
                    new_config[key] = int(text)
                except ValueError:
                    try:
                        new_config[key] = float(text)
                    except ValueError:
                        new_config[key] = text
        return new_config

# ==================== MAIN SCENE (unchanged) ====================

class PerceptionScene(QtWidgets.QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(25, 25, 25)))
        self.nodes = []
        self.edges = []
        self.temp_edge = None
        self.connecting_src = None
        
    def add_node(self, node_class, x=0, y=0):
        sim = node_class()
        node = NodeItem(sim)
        self.addItem(node)
        node.setPos(x, y)
        self.nodes.append(node)
        node.update_display()
        return node
        
    def remove_node(self, node_item):
        if node_item in self.nodes:
            edges_to_remove = [
                e for e in self.edges 
                if e.src.parentItem() == node_item or e.tgt.parentItem() == node_item
            ]
            for edge in edges_to_remove:
                self.remove_edge(edge)
                
            node_item.sim.close() # Calls cleanup in BaseNode subclass
            self.removeItem(node_item)
            self.nodes.remove(node_item)
            
    def remove_edge(self, edge):
        if edge in self.edges:
            self.removeItem(edge)
            self.edges.remove(edge)
            
    def start_connection(self, src_port):
        self.connecting_src = src_port
        self.temp_edge = EdgeItem(src_port)
        self.addItem(self.temp_edge)
        self.temp_edge.update_path()
        
    def finish_connection(self, tgt_port):
        if not self.connecting_src:
            return
        if (self.connecting_src.is_output and not tgt_port.is_output and
            self.connecting_src.port_type == tgt_port.port_type):
            
            if self.connecting_src.parentItem() == tgt_port.parentItem():
                self.cancel_connection()
                return

            edge_exists = any(
                e.src == self.connecting_src and e.tgt == tgt_port for e in self.edges
            )
            if edge_exists:
                self.cancel_connection()
                return
            
            edge = EdgeItem(self.connecting_src, tgt_port)
            self.addItem(edge)
            edge.update_path()
            self.edges.append(edge)
        self.cancel_connection()
        
    def cancel_connection(self):
        if self.temp_edge:
            self.removeItem(self.temp_edge)
        self.temp_edge = None
        self.connecting_src = None
        
    def mousePressEvent(self, ev):
        item = self.itemAt(ev.scenePos(), QtGui.QTransform())
        if isinstance(item, PortItem):
            if item.is_output:
                self.start_connection(item)
                return
            elif self.connecting_src:
                self.finish_connection(item)
                return
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if self.temp_edge and self.connecting_src:
            class FakePort:
                def __init__(self, pos): self._p = pos
                def scenePos(self): return self._p
            self.temp_edge.tgt = FakePort(ev.scenePos())
            self.temp_edge.update_path()
            
    def mouseReleaseEvent(self, ev):
        item = self.itemAt(ev.scenePos(), QtGui.QTransform())
        if isinstance(item, PortItem) and not item.is_output and self.connecting_src:
            self.finish_connection(item)
            return
        if self.connecting_src:
            self.cancel_connection()
        super().mouseReleaseEvent(ev)
        
    def contextMenuEvent(self, ev):
        selected_nodes = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        
        menu = QtWidgets.QMenu()

        if selected_nodes:
            delete_action = menu.addAction(f"Delete Selected Node{'s' if len(selected_nodes) > 1 else ''} ({len(selected_nodes)})")
            delete_action.triggered.connect(lambda: self.delete_selected_nodes())
            
            if len(selected_nodes) == 1 and selected_nodes[0].sim.get_config_options():
                menu.addSeparator()
                config_action = menu.addAction("⚙ Configure Node...")
                config_action.triggered.connect(lambda: self.parent().configure_node(selected_nodes[0]))
        else:
            for name, cls in NODE_TYPES.items():
                action = menu.addAction(f"Add {name}")
                action.triggered.connect(lambda checked, c=cls, p=ev.scenePos(): 
                                         self.add_node(c, x=p.x(), y=p.y()))

        menu.exec(ev.screenPos())

    def delete_selected_nodes(self):
        selected_nodes = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        for node in selected_nodes:
            self.remove_node(node)
            
    def close_all(self):
        for node in self.nodes:
            node.sim.close()

# ==================== MAIN WINDOW (unchanged) ====================

class PerceptionLab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antti's Perception Laboratory")
        self.resize(1400, 900)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.noise_types_list = ["white", "brown", "perlin", "quantum"]
        self.current_noise_index = 0
        
        toolbar = self._create_toolbar()
        layout.addLayout(toolbar)
        
        self.scene = PerceptionScene()
        self.scene.parent = lambda: self 
        
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | 
                                QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        layout.addWidget(self.view, 1)
        
        self.status = QtWidgets.QLabel("Welcome! Add nodes and connect them.")
        self.status.setStyleSheet("color: #aaa; padding: 4px;")
        layout.addWidget(self.status)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step)
        self.is_running = False
        
        self._create_starter_graph()
        self._initialize_toolbar_settings() 
        
    def _find_first_node_of_type(self, node_class):
        """Helper to find the first instance of a node type."""
        for node_item in self.scene.nodes:
            if isinstance(node_item.sim, node_class):
                return node_item
        return None

    def _create_toolbar(self):
        tb = QtWidgets.QHBoxLayout()
        
        add_btn = QtWidgets.QPushButton("➕ Add Node")
        add_menu = QtWidgets.QMenu()
        categories = {}
        for name, cls in NODE_TYPES.items():
            cat = cls.NODE_CATEGORY
            if cat not in categories: categories[cat] = []
            categories[cat].append((name, cls))
        for cat, items in sorted(categories.items()):
            cat_menu = add_menu.addMenu(cat)
            for name, cls in items:
                action = cat_menu.addAction(name)
                action.triggered.connect(lambda checked, c=cls: self.add_node(c))
        add_btn.setMenu(add_menu)
        tb.addWidget(add_btn)
        
        self.run_btn = QtWidgets.QPushButton("▶ Start")
        self.run_btn.clicked.connect(self.toggle_run)
        self.run_btn.setStyleSheet("background: #16a34a; color: white; padding: 6px 12px; font-weight: bold;")
        tb.addWidget(self.run_btn)
        
        clear_btn = QtWidgets.QPushButton("🗑 Clear Edges")
        clear_btn.clicked.connect(self.clear_edges)
        tb.addWidget(clear_btn)
        
        # --- Media Source Controls ---
        tb.addWidget(QtWidgets.QLabel(" |  Media Source:"))
        
        self.media_source_type_combo = QtWidgets.QComboBox()
        self.media_source_type_combo.addItem("Webcam", userData='Webcam')
        self.media_source_type_combo.addItem("Microphone", userData='Microphone')
        self.media_source_type_combo.setToolTip("Select Input Type for first Media Source Node")
        self.media_source_type_combo.currentIndexChanged.connect(self._update_media_source_config)
        tb.addWidget(self.media_source_type_combo)

        self.media_source_device_combo = QtWidgets.QComboBox()
        self.media_source_device_combo.addItem("Device 0", userData=0)
        self.media_source_device_combo.addItem("Device 1", userData=1)
        self.media_source_device_combo.setToolTip("Select Device ID for first Media Source Node")
        self.media_source_device_combo.currentIndexChanged.connect(self._update_media_source_config)
        tb.addWidget(self.media_source_device_combo)
        
        # --- Noise Type Button ---
        tb.addWidget(QtWidgets.QLabel(" |  Noise Type:"))
        self.noise_btn = QtWidgets.QPushButton(f"Noise: {self.noise_types_list[0].capitalize()}")
        self.noise_btn.setToolTip("Cycle noise type for the first Noise Generator Node")
        self.noise_btn.clicked.connect(self._cycle_noise_type)
        tb.addWidget(self.noise_btn)
        
        tb.addWidget(QtWidgets.QLabel("  Coupling:"))
        self.coupling_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.coupling_slider.setRange(0, 100)
        self.coupling_slider.setValue(70)
        self.coupling_slider.setMaximumWidth(150)
        tb.addWidget(self.coupling_slider)
        self.coupling_label = QtWidgets.QLabel("70%")
        self.coupling_slider.valueChanged.connect(
            lambda v: self.coupling_label.setText(f"{v}%"))
        tb.addWidget(self.coupling_label)
        
        tb.addStretch()
        
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #666; font-family: monospace;")
        tb.addWidget(self.fps_label)
        
        return tb
        
    def _cycle_noise_type(self):
        """Cycles the noise type for the first NoiseGeneratorNode."""
        noise_node_item = self._find_first_node_of_type(NoiseGeneratorNode)
        if noise_node_item:
            self.current_noise_index = (self.current_noise_index + 1) % len(self.noise_types_list)
            new_type = self.noise_types_list[self.current_noise_index]
            
            noise_node_item.sim.noise_type = new_type
            
            self.noise_btn.setText(f"Noise: {new_type.capitalize()}")
            noise_node_item.update()
            self.status.setText(f"NoiseGeneratorNode type set to **{new_type.capitalize()}**.")
        else:
            self.status.setText("Warning: No Noise Generator Node found to configure.")

    def _update_media_source_config(self):
        """Updates source type and device ID for the first MediaSourceNode."""
        media_node_item = self._find_first_node_of_type(MediaSourceNode)
        if media_node_item:
            source_type = self.media_source_type_combo.currentData()
            device_id = self.media_source_device_combo.currentData()

            # Apply changes
            media_node_item.sim.source_type = source_type
            media_node_item.sim.device_id = int(device_id)
            
            # Re-setup the stream
            media_node_item.sim.setup_source() 
            media_node_item.update()
            self.status.setText(f"MediaSourceNode updated to **{source_type} (ID: {device_id})**.")
        else:
             self.status.setText("Warning: No Media Source Node found to configure.")
            
    def _initialize_toolbar_settings(self):
        """Sets the toolbar controls to match the initial graph nodes."""
        # Sync Noise Generator button
        noise_node_item = self._find_first_node_of_type(NoiseGeneratorNode)
        if noise_node_item and noise_node_item.sim.noise_type in self.noise_types_list:
            current_type = noise_node_item.sim.noise_type
            self.current_noise_index = self.noise_types_list.index(current_type)
            self.noise_btn.setText(f"Noise: {current_type.capitalize()}")
            
    def _create_starter_graph(self):
        self.scene.nodes = []
        self.scene.edges = []
        self.scene.clear()
        
        media_source = self.scene.add_node(MediaSourceNode, x=50, y=100)
        signal_proc = self.scene.add_node(SignalProcessorNode, x=300, y=300) # Added processor
        fft = self.scene.add_node(FFTCochleaNode, x=550, y=100)
        speaker = self.scene.add_node(SpeakerOutputNode, x=800, y=300)
        monitor = self.scene.add_node(SignalMonitorNode, x=800, y=100)
        display = self.scene.add_node(ImageDisplayNode, x=800, y=500) # Added display for FFT image
        
        self.connect_nodes(media_source, 'signal', signal_proc, 'input_signal')
        self.connect_nodes(signal_proc, 'output_signal', speaker, 'signal')
        self.connect_nodes(media_source, 'image', fft, 'image')
        self.connect_nodes(signal_proc, 'output_signal', monitor, 'signal')
        self.connect_nodes(fft, 'image', display, 'image') # FFT now outputs image
        
        for e in self.scene.edges:
            e.update_path()

        self.status.setText("Starter graph created! Press **Start** to run. Use **DEL** or **Right-Click** to delete/configure nodes.")
        
    def connect_nodes(self, src_node_item, src_port_name, tgt_node_item, tgt_port_name):
        src_port = src_node_item.out_ports[src_port_name]
        tgt_port = tgt_node_item.in_ports[tgt_port_name]
        edge = EdgeItem(src_port, tgt_port)
        self.scene.addItem(edge)
        self.scene.edges.append(edge)

    def add_node(self, node_class):
        view_center = self.view.mapToScene(self.view.viewport().rect().center())
        node = self.scene.add_node(node_class, x=view_center.x()-100, y=view_center.y()-80)
        self.status.setText(f"Added {node.sim.node_title}")
        
    def configure_node(self, node_item):
        dialog = NodeConfigDialog(node_item, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_config = dialog.get_new_config()
            
            # Apply changes
            for key, value in new_config.items():
                # Special handling for int/float fields based on node type if necessary
                if key in ['device_id', 'sample_rate', 'wipe_width']:
                     value = int(value) if value is not None else None
                elif key in ['factor', 'speed', 'threshold']:
                     value = float(value) if value is not None else None

                setattr(node_item.sim, key, value)
                
            # Re-setup resources
            if isinstance(node_item.sim, SpeakerOutputNode):
                node_item.sim.open_stream()
            elif isinstance(node_item.sim, MediaSourceNode):
                node_item.sim.setup_source()
            
            node_item.update()
            self.status.setText(f"Configured {node_item.sim.node_title}")
        
    def toggle_run(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.run_btn.setText("⏸ Stop")
            self.run_btn.setStyleSheet("background: #dc2626; color: white; padding: 6px 12px; font-weight: bold;")
            self.timer.start(33)
            self.status.setText("Running...")
            self.last_time = QtCore.QTime.currentTime()
            self.frame_count = 0
        else:
            self.run_btn.setText("▶ Start")
            self.run_btn.setStyleSheet("background: #16a34a; color: white; padding: 6px 12px; font-weight: bold;")
            self.timer.stop()
            self.status.setText("Stopped")
            
    def clear_edges(self):
        for edge in list(self.scene.edges):
            self.scene.remove_edge(edge)
        self.status.setText("Cleared all edges")
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Delete or event.key() == QtCore.Qt.Key.Key_Backspace:
            self.scene.delete_selected_nodes()
            self.status.setText("Deleted selected nodes.")
            return
        super().keyPressEvent(event)
        
    def step(self):
        
        for node in self.scene.nodes:
            node.sim.pre_step()
            
        node_map = {n: n for n in self.scene.nodes}
        coupling = self.coupling_slider.value() / 100.0
        
        for edge in self.scene.edges:
            src_node = edge.src.parentItem()
            tgt_node = edge.tgt.parentItem()
            
            if src_node not in node_map or tgt_node not in node_map:
                continue
                
            output = src_node.sim.get_output(edge.src.name)
            if output is None:
                continue
                
            tgt_node.sim.set_input(edge.tgt.name, output, edge.src.port_type, coupling)
            
            if edge.src.port_type == 'signal':
                if isinstance(output, (float, int)):
                    edge.effect_val = abs(float(output) * coupling)
                elif isinstance(output, np.ndarray) and output.size == 1:
                    edge.effect_val = abs(float(output.flat[0]) * coupling)
                else:
                    edge.effect_val = 0.5
            elif edge.src.port_type == 'spectrum' or edge.src.port_type == 'image' or edge.src.port_type == 'complex_spectrum':
                edge.effect_val = 0.8
                if isinstance(output, np.ndarray) and output.size > 0:
                     edge.effect_val += np.mean(np.abs(output)) * 0.1
            else:
                edge.effect_val = 0.5
            edge.update_path()
            
        for node in self.scene.nodes:
            node.sim.step()
            node.update_display()
            
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = QtCore.QTime.currentTime()
            elapsed = self.last_time.msecsTo(current_time) / 1000.0
            if elapsed > 0:
                fps = 30.0 / elapsed
                self.fps_label.setText(f"FPS: {fps:.1f}")
            self.last_time = current_time
            
    def closeEvent(self, event):
        self.timer.stop()
        self.scene.close_all()
        super().closeEvent(event)

# ==================== APPLICATION ENTRY ====================

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 80, 80))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(100, 150, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(100, 150, 255))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(20, 20, 20))
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QPushButton {
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            background: #3a3a3a;
            color: #ddd;
        }
        QPushButton:hover {
            background: #4a4a4a;
        }
        QPushButton:pressed {
            background: #2a2a2a;
        }
        QPushButton::menu-indicator {
            width: 0px;
        }
        QMenu {
            background: #2a2a2a;
            border: 1px solid #444;
        }
        QMenu::item {
            padding: 6px 20px;
        }
        QMenu::item:selected {
            background: #3a5a8a;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #3a3a3a;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #6495ed;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #7ab5ff;
        }
        QLineEdit, QComboBox {
            background: #3a3a3a;
            border: 1px solid #555;
            padding: 2px;
            color: #ddd;
            border-radius: 4px;
            height: 24px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left-width: 1px;
            border-left-color: #555;
            border-left-style: solid;
        }
    """)
    
    window = PerceptionLab()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()