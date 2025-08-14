# sigils_dictionary.py
"""
Sigils Dictionary - Unicode Symbol Library for Glyphs

This module maintains the library of available Unicode symbols for use in glyphs.
Once a sigil is used, it becomes unavailable to prevent reuse.
Organized by categories for appropriate selection.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger('SigilsDictionary')

class SigilsDictionary:
    """Manages available Unicode symbols for glyph creation"""
    
    def __init__(self, save_path: str = "shared/data/sigils_availability.json"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Complete Unicode symbol library organized by category
        self.available_sigils = {
            'sacred_geometry': [
                '⚹', '⚪', '⚫', '○', '●', '◯', '◉', '◎', '⊙', '⊕', '⊖', '⊗', '⊘', '⊚', '⊛', '⊜', '⊝',
                '△', '▲', '▽', '▼', '◁', '◀', '▷', '▶', '◇', '◆', '◈', '◉', '◊', '♦', '⬟', '⬠', '⬡',
                '⬢', '⬣', '⟐', '⟑', '⟒', '⟓', '⟔', '⟕', '⟖', '⟗', '⟘', '⟙', '⟚', '⟛', '⟜', '⟝', '⟞'
            ],
            
            'spiritual': [
                '☀', '☁', '☂', '☃', '☄', '★', '☆', '☇', '☈', '☉', '☊', '☋', '☌', '☍', '☎', '☏',
                '☐', '☑', '☒', '☓', '☔', '☕', '☖', '☗', '☘', '☙', '☚', '☛', '☜', '☝', '☞', '☟',
                '☠', '☡', '☢', '☣', '☤', '☥', '☦', '☧', '☨', '☩', '☪', '☫', '☬', '☭', '☮', '☯',
                '☰', '☱', '☲', '☳', '☴', '☵', '☶', '☷', '☸', '☹', '☺', '☻', '☼', '☽', '☾', '☿'
            ],
            
            'elemental': [
                '♁', '♂', '♃', '♄', '♅', '♆', '♇', '♈', '♉', '♊', '♋', '♌', '♍', '♎', '♏', '♐',
                '♑', '♒', '♓', '♔', '♕', '♖', '♗', '♘', '♙', '♚', '♛', '♜', '♝', '♞', '♟', '♠',
                '♡', '♢', '♣', '♤', '♥', '♦', '♧', '♨', '♩', '♪', '♫', '♬', '♭', '♮', '♯', '♰'
            ],
            
            'mystical': [
                '⚀', '⚁', '⚂', '⚃', '⚄', '⚅', '⚆', '⚇', '⚈', '⚉', '⚊', '⚋', '⚌', '⚍', '⚎', '⚏',
                '⚐', '⚑', '⚒', '⚓', '⚔', '⚕', '⚖', '⚗', '⚘', '⚙', '⚚', '⚛', '⚜', '⚝', '⚞', '⚟',
                '⚠', '⚡', '⚢', '⚣', '⚤', '⚥', '⚦', '⚧', '⚨', '⚩', '⚪', '⚫', '⚬', '⚭', '⚮', '⚯'
            ],
            
            'cosmic': [
                '⊰', '⊱', '⊲', '⊳', '⊴', '⊵', '⊶', '⊷', '⊸', '⊹', '⊺', '⊻', '⊼', '⊽', '⊾', '⊿',
                '⋀', '⋁', '⋂', '⋃', '⋄', '⋅', '⋆', '⋇', '⋈', '⋉', '⋊', '⋋', '⋌', '⋍', '⋎', '⋏',
                '⋐', '⋑', '⋒', '⋓', '⋔', '⋕', '⋖', '⋗', '⋘', '⋙', '⋚', '⋛', '⋜', '⋝', '⋞', '⋟'
            ],
            
            'alchemical': [
                '🜀', '🜁', '🜂', '🜃', '🜄', '🜅', '🜆', '🜇', '🜈', '🜉', '🜊', '🜋', '🜌', '🜍', '🜎', '🜏',
                '🜐', '🜑', '🜒', '🜓', '🜔', '🜕', '🜖', '🜗', '🜘', '🜙', '🜚', '🜛', '🜜', '🜝', '🜞', '🜟',
                '🜠', '🜡', '🜢', '🜣', '🜤', '🜥', '🜦', '🜧', '🜨', '🜩', '🜪', '🜫', '🜬', '🜭', '🜮', '🜯'
            ],
            
            'runic': [
                'ᚠ', 'ᚡ', 'ᚢ', 'ᚣ', 'ᚤ', 'ᚥ', 'ᚦ', 'ᚧ', 'ᚨ', 'ᚩ', 'ᚪ', 'ᚫ', 'ᚬ', 'ᚭ', 'ᚮ', 'ᚯ',
                'ᚰ', 'ᚱ', 'ᚲ', 'ᚳ', 'ᚴ', 'ᚵ', 'ᚶ', 'ᚷ', 'ᚸ', 'ᚹ', 'ᚺ', 'ᚻ', 'ᚼ', 'ᚽ', 'ᚾ', 'ᚿ',
                'ᛀ', 'ᛁ', 'ᛂ', 'ᛃ', 'ᛄ', 'ᛅ', 'ᛆ', 'ᛇ', 'ᛈ', 'ᛉ', 'ᛊ', 'ᛋ', 'ᛌ', 'ᛍ', 'ᛎ', 'ᛏ'
            ],
            
            'mathematical': [
                '∀', '∁', '∂', '∃', '∄', '∅', '∆', '∇', '∈', '∉', '∊', '∋', '∌', '∍', '∎', '∏',
                '∐', '∑', '−', '∓', '∔', '∕', '∖', '∗', '∘', '∙', '√', '∛', '∜', '∝', '∞', '∟',
                '∠', '∡', '∢', '∣', '∤', '∥', '∦', '∧', '∨', '∩', '∪', '∫', '∬', '∭', '∮', '∯'
            ],
        }
        
        # Load existing usage state
        self._load_availability_state()
    
    def _load_availability_state(self):
        """Load the current availability state from file"""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    used_sigils = json.load(f)
                
                # Remove used sigils from available ones
                for category, sigils in used_sigils.items():
                    if category in self.available_sigils:
                        for sigil in sigils:
                            if sigil in self.available_sigils[category]:
                                self.available_sigils[category].remove(sigil)
                
                logger.info(f"Loaded sigil availability state from {self.save_path}")
            except Exception as e:
                logger.warning(f"Could not load sigil state: {e}")
        else:
            logger.info("No existing sigil state found, starting fresh")
    
    def _save_availability_state(self, used_sigils: Dict[str, List[str]]):
        """Save the current used sigils to file"""
        try:
            # Load existing used sigils
            existing_used = {}
            if self.save_path.exists():
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    existing_used = json.load(f)
            
            # Merge with new used sigils
            for category, sigils in used_sigils.items():
                if category not in existing_used:
                    existing_used[category] = []
                existing_used[category].extend(sigils)
                # Remove duplicates
                existing_used[category] = list(set(existing_used[category]))
            
            # Save updated state
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(existing_used, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved sigil availability state to {self.save_path}")
        except Exception as e:
            logger.error(f"Could not save sigil state: {e}")
    
    def get_available_sigils(self, category: str) -> List[str]:
        """Get all available sigils for a category"""
        return self.available_sigils.get(category, []).copy()
    
    def get_sigil(self, category: str, specific_sigil: Optional[str] = None) -> Optional[str]:
        """
        Get a sigil from the specified category and mark it as used
        
        Args:
            category: Category to select from
            specific_sigil: Specific sigil to use (if available)
            
        Returns:
            Selected sigil or None if not available
        """
        if category not in self.available_sigils:
            logger.warning(f"Category {category} not found in sigils dictionary")
            return None
        
        available = self.available_sigils[category]
        
        if not available:
            logger.warning(f"No available sigils in category {category}")
            return None
        
        # Select specific sigil or random one
        if specific_sigil and specific_sigil in available:
            selected = specific_sigil
        else:
            import random
            selected = random.choice(available)
        
        # Remove from available and mark as used
        self.available_sigils[category].remove(selected)
        self._save_availability_state({category: [selected]})
        
        logger.info(f"Selected sigil '{selected}' from category '{category}'")
        return selected
    
    def reserve_sigils(self, reservations: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Reserve multiple sigils at once
        
        Args:
            reservations: Dict of {category: [sigil1, sigil2, ...]}
            
        Returns:
            Dict of {purpose: sigil} for successful reservations
        """
        reserved = {}
        used_sigils = {}
        
        for category, sigil_list in reservations.items():
            used_sigils[category] = []
            for sigil in sigil_list:
                if sigil in self.available_sigils.get(category, []):
                    self.available_sigils[category].remove(sigil)
                    reserved[f"{category}_{len(reserved)}"] = sigil
                    used_sigils[category].append(sigil)
        
        if used_sigils:
            self._save_availability_state(used_sigils)
        
        return reserved
    
    def get_category_count(self, category: str) -> int:
        """Get count of available sigils in category"""
        return len(self.available_sigils.get(category, []))
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.available_sigils.keys())
    
    def reset_availability(self):
        """Reset all sigils to available (use with caution)"""
        if self.save_path.exists():
            self.save_path.unlink()
        self._load_availability_state()
        logger.warning("Reset all sigil availability - all sigils are now available again")

# Factory function
def create_sigils_dictionary() -> SigilsDictionary:
    """Create a new sigils dictionary instance"""
    return SigilsDictionary()