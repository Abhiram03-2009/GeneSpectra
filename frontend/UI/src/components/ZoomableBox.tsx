import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FiZoomIn, FiX } from 'react-icons/fi';
import { motion, AnimatePresence } from 'framer-motion';

interface ZoomableBoxProps {
  title: string;
  description: string;
  calculation: string;
  children: React.ReactNode;
}

const ZoomableBox: React.FC<ZoomableBoxProps> = ({ title, description, calculation, children }) => {
  const [isPopupOpen, setIsPopupOpen] = useState(false);
  const popupRef = useRef<HTMLDivElement>(null);

  const openPopup = useCallback(() => {
    setIsPopupOpen(true);
  }, []);

  const closePopup = useCallback(() => {
    setIsPopupOpen(false);
  }, []);

  // Close popup if clicked outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(event.target as Node)) {
        closePopup();
      }
    };
    if (isPopupOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    } else {
      document.removeEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isPopupOpen, closePopup]);

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="hud-card relative overflow-hidden rounded-3xl border border-cyan-600/40 backdrop-blur-lg bg-gradient-to-br from-[#0a1e33]/90 to-[#142b43]/90 shadow-2xl p-8 hover:border-cyan-300 transition-all duration-300 group flex flex-col items-center justify-center"
    >
      <h3 className="text-2xl font-bold text-white mb-6 text-center tracking-tight">{title}</h3>
      <button
        onClick={openPopup}
        className="absolute top-4 right-4 z-20 p-2 rounded-full bg-black/50 hover:bg-black/70 text-cyan-300 hover:text-white transition-all duration-200"
        aria-label={`Zoom into ${title}`}
      >
        <FiZoomIn size={20} />
      </button>

      <div className="flex-grow w-full h-full flex items-center justify-center">
        {children}
      </div>

      <AnimatePresence>
        {isPopupOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-[100] outline-none"
            onClick={closePopup} // Close on overlay click
          >
            <motion.div
              ref={popupRef}
              className="relative hud-card p-6 rounded-3xl border border-cyan-400/50 shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col"
              onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside popup
            >
              <button
                onClick={closePopup}
                className="absolute top-4 right-4 z-10 p-2 rounded-full bg-black/50 hover:bg-black/70 text-cyan-300 hover:text-white transition-all duration-200"
                aria-label="Close"
              >
                <FiX size={24} />
              </button>
              <h3 className="text-3xl font-bold text-cyan-300 mb-4 text-center select-none">{title} - Detailed View</h3>
              <div className="flex-grow overflow-y-auto custom-scrollbar pr-2 mb-4">
                <div className="w-full h-auto object-contain rounded-md border border-gray-700 mb-4 flex items-center justify-center">
                    {children} {/* Display the children (image/chart) in the popup */}
                </div>
                <div className="text-white/80 text-base space-y-3">
                  <p><strong className="text-cyan-300">Description:</strong> {description}</p>
                  <p><strong className="text-cyan-300">Calculation:</strong> {calculation}</p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ZoomableBox;



