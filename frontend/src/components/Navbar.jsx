import React from 'react';
import { motion } from 'framer-motion';
import { Menu, X, LogOut } from 'lucide-react';

const NavBar = ({ isMenuOpen, setIsMenuOpen, onLogout }) => {
  return (
    <nav className="backdrop-blur-md bg-black/10 border-b border-white/5 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto flex justify-between items-center p-4">
        <span className="text-lg font-light">gym<span className="text-purple-400">tracker</span></span>
        
        <div className="flex items-center space-x-4">
          {onLogout && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onLogout}
              className="hidden md:flex items-center space-x-2 bg-red-500/10 hover:bg-red-500/20 px-3 py-1.5 rounded-lg text-sm text-red-400"
            >
              <LogOut size={16} />
              <span>Logout</span>
            </motion.button>
          )}
          
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="lg:hidden p-1.5 rounded-lg bg-white/5"
          >
            {isMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </motion.button>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;