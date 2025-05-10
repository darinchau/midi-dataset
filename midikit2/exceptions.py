class MidiDecodeError(ValueError):
    """Exception raised when decoding a MIDI file fails, which indicates either a faulty file or not a MIDI file"""
    pass


class MidiEOFError(MidiDecodeError, EOFError):
    """Special subclass of MidiDecodeError for EOF errors"""
    pass
