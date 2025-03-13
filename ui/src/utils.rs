use std::{fs, path::PathBuf, str::FromStr};

use ::image::{DynamicImage, GenericImage, Rgba, open};
use common::*;
use iced::{Point, widget::image};

use crate::State;

pub fn key_to_string(named: iced::keyboard::key::Named) -> String {
    match named {
        iced::keyboard::key::Named::Alt => "Alt".to_string(),
        iced::keyboard::key::Named::AltGraph => "AltGraph".to_string(),
        iced::keyboard::key::Named::CapsLock => "CapsLock".to_string(),
        iced::keyboard::key::Named::Control => "Control".to_string(),
        iced::keyboard::key::Named::Fn => "Fn".to_string(),
        iced::keyboard::key::Named::FnLock => "FnLock".to_string(),
        iced::keyboard::key::Named::NumLock => "NumLock".to_string(),
        iced::keyboard::key::Named::ScrollLock => "ScrollLock".to_string(),
        iced::keyboard::key::Named::Shift => "Shift".to_string(),
        iced::keyboard::key::Named::Symbol => "Symbol".to_string(),
        iced::keyboard::key::Named::SymbolLock => "SymbolLock".to_string(),
        iced::keyboard::key::Named::Meta => "Meta".to_string(),
        iced::keyboard::key::Named::Hyper => "Hyper".to_string(),
        iced::keyboard::key::Named::Super => "Super".to_string(),
        iced::keyboard::key::Named::Enter => "Enter".to_string(),
        iced::keyboard::key::Named::Tab => "Tab".to_string(),
        iced::keyboard::key::Named::Space => "Space".to_string(),
        iced::keyboard::key::Named::ArrowDown => "ArrowDown".to_string(),
        iced::keyboard::key::Named::ArrowLeft => "ArrowLeft".to_string(),
        iced::keyboard::key::Named::ArrowRight => "ArrowRight".to_string(),
        iced::keyboard::key::Named::ArrowUp => "ArrowUp".to_string(),
        iced::keyboard::key::Named::End => "End".to_string(),
        iced::keyboard::key::Named::Home => "Home".to_string(),
        iced::keyboard::key::Named::PageDown => "PageDown".to_string(),
        iced::keyboard::key::Named::PageUp => "PageUp".to_string(),
        iced::keyboard::key::Named::Backspace => "Backspace".to_string(),
        iced::keyboard::key::Named::Clear => "Clear".to_string(),
        iced::keyboard::key::Named::Copy => "Copy".to_string(),
        iced::keyboard::key::Named::CrSel => "CrSel".to_string(),
        iced::keyboard::key::Named::Cut => "Cut".to_string(),
        iced::keyboard::key::Named::Delete => "Delete".to_string(),
        iced::keyboard::key::Named::EraseEof => "EraseEof".to_string(),
        iced::keyboard::key::Named::ExSel => "ExSel".to_string(),
        iced::keyboard::key::Named::Insert => "Insert".to_string(),
        iced::keyboard::key::Named::Paste => "Paste".to_string(),
        iced::keyboard::key::Named::Redo => "Redo".to_string(),
        iced::keyboard::key::Named::Undo => "Undo".to_string(),
        iced::keyboard::key::Named::Accept => "Accept".to_string(),
        iced::keyboard::key::Named::Again => "Again".to_string(),
        iced::keyboard::key::Named::Attn => "Attn".to_string(),
        iced::keyboard::key::Named::Cancel => "Cancel".to_string(),
        iced::keyboard::key::Named::ContextMenu => "ContextMenu".to_string(),
        iced::keyboard::key::Named::Escape => "Escape".to_string(),
        iced::keyboard::key::Named::Execute => "Execute".to_string(),
        iced::keyboard::key::Named::Find => "Find".to_string(),
        iced::keyboard::key::Named::Help => "Help".to_string(),
        iced::keyboard::key::Named::Pause => "Pause".to_string(),
        iced::keyboard::key::Named::Play => "Play".to_string(),
        iced::keyboard::key::Named::Props => "Props".to_string(),
        iced::keyboard::key::Named::Select => "Select".to_string(),
        iced::keyboard::key::Named::ZoomIn => "ZoomIn".to_string(),
        iced::keyboard::key::Named::ZoomOut => "ZoomOut".to_string(),
        iced::keyboard::key::Named::BrightnessDown => "BrightnessDown".to_string(),
        iced::keyboard::key::Named::BrightnessUp => "BrightnessUp".to_string(),
        iced::keyboard::key::Named::Eject => "Eject".to_string(),
        iced::keyboard::key::Named::LogOff => "LogOff".to_string(),
        iced::keyboard::key::Named::Power => "Power".to_string(),
        iced::keyboard::key::Named::PowerOff => "PowerOff".to_string(),
        iced::keyboard::key::Named::PrintScreen => "PrintScreen".to_string(),
        iced::keyboard::key::Named::Hibernate => "Hibernate".to_string(),
        iced::keyboard::key::Named::Standby => "Standby".to_string(),
        iced::keyboard::key::Named::WakeUp => "WakeUp".to_string(),
        iced::keyboard::key::Named::AllCandidates => "AllCandidates".to_string(),
        iced::keyboard::key::Named::Alphanumeric => "Alphanumeric".to_string(),
        iced::keyboard::key::Named::CodeInput => "CodeInput".to_string(),
        iced::keyboard::key::Named::Compose => "Compose".to_string(),
        iced::keyboard::key::Named::Convert => "Convert".to_string(),
        iced::keyboard::key::Named::FinalMode => "FinalMode".to_string(),
        iced::keyboard::key::Named::GroupFirst => "GroupFirst".to_string(),
        iced::keyboard::key::Named::GroupLast => "GroupLast".to_string(),
        iced::keyboard::key::Named::GroupNext => "GroupNext".to_string(),
        iced::keyboard::key::Named::GroupPrevious => "GroupPrevious".to_string(),
        iced::keyboard::key::Named::ModeChange => "ModeChange".to_string(),
        iced::keyboard::key::Named::NextCandidate => "NextCandidate".to_string(),
        iced::keyboard::key::Named::NonConvert => "NonConvert".to_string(),
        iced::keyboard::key::Named::PreviousCandidate => "PreviousCandidate".to_string(),
        iced::keyboard::key::Named::Process => "Process".to_string(),
        iced::keyboard::key::Named::SingleCandidate => "SingleCandidate".to_string(),
        iced::keyboard::key::Named::HangulMode => "HangulMode".to_string(),
        iced::keyboard::key::Named::HanjaMode => "HanjaMode".to_string(),
        iced::keyboard::key::Named::JunjaMode => "JunjaMode".to_string(),
        iced::keyboard::key::Named::Eisu => "Eisu".to_string(),
        iced::keyboard::key::Named::Hankaku => "Hankaku".to_string(),
        iced::keyboard::key::Named::Hiragana => "Hiragana".to_string(),
        iced::keyboard::key::Named::HiraganaKatakana => "HiraganaKatakana".to_string(),
        iced::keyboard::key::Named::KanaMode => "KanaMode".to_string(),
        iced::keyboard::key::Named::KanjiMode => "KanjiMode".to_string(),
        iced::keyboard::key::Named::Katakana => "Katakana".to_string(),
        iced::keyboard::key::Named::Romaji => "Romaji".to_string(),
        iced::keyboard::key::Named::Zenkaku => "Zenkaku".to_string(),
        iced::keyboard::key::Named::ZenkakuHankaku => "ZenkakuHankaku".to_string(),
        iced::keyboard::key::Named::Soft1 => "Soft1".to_string(),
        iced::keyboard::key::Named::Soft2 => "Soft2".to_string(),
        iced::keyboard::key::Named::Soft3 => "Soft3".to_string(),
        iced::keyboard::key::Named::Soft4 => "Soft4".to_string(),
        iced::keyboard::key::Named::ChannelDown => "ChannelDown".to_string(),
        iced::keyboard::key::Named::ChannelUp => "ChannelUp".to_string(),
        iced::keyboard::key::Named::Close => "Close".to_string(),
        iced::keyboard::key::Named::MailForward => "MailForward".to_string(),
        iced::keyboard::key::Named::MailReply => "MailReply".to_string(),
        iced::keyboard::key::Named::MailSend => "MailSend".to_string(),
        iced::keyboard::key::Named::MediaClose => "MediaClose".to_string(),
        iced::keyboard::key::Named::MediaFastForward => "MediaFastForward".to_string(),
        iced::keyboard::key::Named::MediaPause => "MediaPause".to_string(),
        iced::keyboard::key::Named::MediaPlay => "MediaPlay".to_string(),
        iced::keyboard::key::Named::MediaPlayPause => "MediaPlayPause".to_string(),
        iced::keyboard::key::Named::MediaRecord => "MediaRecord".to_string(),
        iced::keyboard::key::Named::MediaRewind => "MediaRewind".to_string(),
        iced::keyboard::key::Named::MediaStop => "MediaStop".to_string(),
        iced::keyboard::key::Named::MediaTrackNext => "MediaTrackNext".to_string(),
        iced::keyboard::key::Named::MediaTrackPrevious => "MediaTrackPrevious".to_string(),

        iced::keyboard::key::Named::New => "New".to_string(),
        iced::keyboard::key::Named::Open => "Open".to_string(),
        iced::keyboard::key::Named::Print => "Print".to_string(),
        iced::keyboard::key::Named::Save => "Save".to_string(),
        iced::keyboard::key::Named::SpellCheck => "SpellCheck".to_string(),
        iced::keyboard::key::Named::Key11 => "Key11".to_string(),
        iced::keyboard::key::Named::Key12 => "Key12".to_string(),
        iced::keyboard::key::Named::AudioBalanceLeft => "AudioBalanceLeft".to_string(),
        iced::keyboard::key::Named::AudioBalanceRight => "AudioBalanceRight".to_string(),
        iced::keyboard::key::Named::AudioBassBoostDown => "AudioBassBoostDown".to_string(),
        iced::keyboard::key::Named::AudioBassBoostToggle => "AudioBassBoostToggle".to_string(),
        iced::keyboard::key::Named::AudioBassBoostUp => "AudioBassBoostUp".to_string(),
        iced::keyboard::key::Named::AudioFaderFront => "AudioFaderFront".to_string(),
        iced::keyboard::key::Named::AudioFaderRear => "AudioFaderRear".to_string(),
        iced::keyboard::key::Named::AudioSurroundModeNext => "AudioSurroundModeNext".to_string(),
        iced::keyboard::key::Named::AudioTrebleDown => "AudioTrebleDown".to_string(),
        iced::keyboard::key::Named::AudioTrebleUp => "AudioTrebleUp".to_string(),
        iced::keyboard::key::Named::AudioVolumeDown => "AudioVolumeDown".to_string(),
        iced::keyboard::key::Named::AudioVolumeUp => "AudioVolumeUp".to_string(),
        iced::keyboard::key::Named::AudioVolumeMute => "AudioVolumeMute".to_string(),
        iced::keyboard::key::Named::MicrophoneToggle => "MicrophoneToggle".to_string(),
        iced::keyboard::key::Named::MicrophoneVolumeDown => "MicrophoneVolumeDown".to_string(),
        iced::keyboard::key::Named::MicrophoneVolumeUp => "MicrophoneVolumeUp".to_string(),
        iced::keyboard::key::Named::MicrophoneVolumeMute => "MicrophoneVolumeMute".to_string(),
        iced::keyboard::key::Named::SpeechCorrectionList => "SpeechCorrectionList".to_string(),
        iced::keyboard::key::Named::SpeechInputToggle => "SpeechInputToggle".to_string(),
        iced::keyboard::key::Named::LaunchApplication1 => "LaunchApplication1".to_string(),
        iced::keyboard::key::Named::LaunchApplication2 => "LaunchApplication2".to_string(),
        iced::keyboard::key::Named::LaunchCalendar => "LaunchCalendar".to_string(),
        iced::keyboard::key::Named::LaunchContacts => "LaunchContacts".to_string(),
        iced::keyboard::key::Named::LaunchMail => "LaunchMail".to_string(),
        iced::keyboard::key::Named::LaunchMediaPlayer => "LaunchMediaPlayer".to_string(),
        iced::keyboard::key::Named::LaunchMusicPlayer => "LaunchMusicPlayer".to_string(),
        iced::keyboard::key::Named::LaunchPhone => "LaunchPhone".to_string(),
        iced::keyboard::key::Named::LaunchScreenSaver => "LaunchScreenSaver".to_string(),
        iced::keyboard::key::Named::LaunchSpreadsheet => "LaunchSpreadsheet".to_string(),
        iced::keyboard::key::Named::LaunchWebBrowser => "LaunchWebBrowser".to_string(),
        iced::keyboard::key::Named::LaunchWebCam => "LaunchWebCam".to_string(),
        iced::keyboard::key::Named::LaunchWordProcessor => "LaunchWordProcessor".to_string(),
        iced::keyboard::key::Named::BrowserBack => "BrowserBack".to_string(),
        iced::keyboard::key::Named::BrowserFavorites => "BrowserFavorites".to_string(),
        iced::keyboard::key::Named::BrowserForward => "BrowserForward".to_string(),
        iced::keyboard::key::Named::BrowserHome => "BrowserHome".to_string(),
        iced::keyboard::key::Named::BrowserRefresh => "BrowserRefresh".to_string(),
        iced::keyboard::key::Named::BrowserSearch => "BrowserSearch".to_string(),
        iced::keyboard::key::Named::BrowserStop => "BrowserStop".to_string(),
        iced::keyboard::key::Named::AppSwitch => "AppSwitch".to_string(),
        iced::keyboard::key::Named::Call => "Call".to_string(),
        iced::keyboard::key::Named::Camera => "Camera".to_string(),
        iced::keyboard::key::Named::CameraFocus => "CameraFocus".to_string(),
        iced::keyboard::key::Named::EndCall => "EndCall".to_string(),
        iced::keyboard::key::Named::GoBack => "GoBack".to_string(),
        iced::keyboard::key::Named::GoHome => "GoHome".to_string(),
        iced::keyboard::key::Named::HeadsetHook => "HeadsetHook".to_string(),
        iced::keyboard::key::Named::LastNumberRedial => "LastNumberRedial".to_string(),
        iced::keyboard::key::Named::Notification => "Notification".to_string(),
        iced::keyboard::key::Named::MannerMode => "MannerMode".to_string(),
        iced::keyboard::key::Named::VoiceDial => "VoiceDial".to_string(),
        iced::keyboard::key::Named::TV => "TV".to_string(),
        iced::keyboard::key::Named::TV3DMode => "TV3DMode".to_string(),
        iced::keyboard::key::Named::TVAntennaCable => "TVAntennaCable".to_string(),
        iced::keyboard::key::Named::TVAudioDescription => "TVAudioDescription".to_string(),
        iced::keyboard::key::Named::TVAudioDescriptionMixDown => {
            "TVAudioDescriptionMixDown".to_string()
        }
        iced::keyboard::key::Named::TVAudioDescriptionMixUp => {
            "TVAudioDescriptionMixUp".to_string()
        }
        iced::keyboard::key::Named::TVContentsMenu => "TVContentsMenu".to_string(),
        iced::keyboard::key::Named::TVDataService => "TVDataService".to_string(),
        iced::keyboard::key::Named::TVInput => "TVInput".to_string(),
        iced::keyboard::key::Named::TVInputComponent1 => "TVInputComponent1".to_string(),
        iced::keyboard::key::Named::TVInputComponent2 => "TVInputComponent2".to_string(),
        iced::keyboard::key::Named::TVInputComposite1 => "TVInputComposite1".to_string(),
        iced::keyboard::key::Named::TVInputComposite2 => "TVInputComposite2".to_string(),
        iced::keyboard::key::Named::TVInputHDMI1 => "TVInputHDMI1".to_string(),
        iced::keyboard::key::Named::TVInputHDMI2 => "TVInputHDMI2".to_string(),
        iced::keyboard::key::Named::TVInputHDMI3 => "TVInputHDMI3".to_string(),
        iced::keyboard::key::Named::TVInputHDMI4 => "TVInputHDMI4".to_string(),
        iced::keyboard::key::Named::TVInputVGA1 => "TVInputVGA1".to_string(),
        iced::keyboard::key::Named::TVMediaContext => "TVMediaContext".to_string(),
        iced::keyboard::key::Named::TVNetwork => "TVNetwork".to_string(),
        iced::keyboard::key::Named::TVNumberEntry => "TVNumberEntry".to_string(),
        iced::keyboard::key::Named::TVPower => "TVPower".to_string(),
        iced::keyboard::key::Named::TVRadioService => "TVRadioService".to_string(),
        iced::keyboard::key::Named::TVSatellite => "TVSatellite".to_string(),
        iced::keyboard::key::Named::TVSatelliteBS => "TVSatelliteBS".to_string(),
        iced::keyboard::key::Named::TVSatelliteCS => "TVSatelliteCS".to_string(),
        iced::keyboard::key::Named::TVSatelliteToggle => "TVSatelliteToggle".to_string(),
        iced::keyboard::key::Named::TVTerrestrialAnalog => "TVTerrestrialAnalog".to_string(),
        iced::keyboard::key::Named::TVTerrestrialDigital => "TVTerrestrialDigital".to_string(),
        iced::keyboard::key::Named::TVTimer => "TVTimer".to_string(),
        iced::keyboard::key::Named::AVRInput => "AVRInput".to_string(),
        iced::keyboard::key::Named::AVRPower => "AVRPower".to_string(),
        iced::keyboard::key::Named::ColorF0Red => "ColorF0Red".to_string(),
        iced::keyboard::key::Named::ColorF1Green => "ColorF1Green".to_string(),
        iced::keyboard::key::Named::ColorF2Yellow => "ColorF2Yellow".to_string(),
        iced::keyboard::key::Named::ColorF3Blue => "ColorF3Blue".to_string(),
        iced::keyboard::key::Named::ColorF4Grey => "ColorF4Grey".to_string(),
        iced::keyboard::key::Named::ColorF5Brown => "ColorF5Brown".to_string(),
        iced::keyboard::key::Named::ClosedCaptionToggle => "ClosedCaptionToggle".to_string(),
        iced::keyboard::key::Named::Dimmer => "Dimmer".to_string(),
        iced::keyboard::key::Named::DisplaySwap => "DisplaySwap".to_string(),
        iced::keyboard::key::Named::DVR => "DVR".to_string(),
        iced::keyboard::key::Named::Exit => "Exit".to_string(),
        iced::keyboard::key::Named::FavoriteClear0 => "FavoriteClear0".to_string(),
        iced::keyboard::key::Named::FavoriteClear1 => "FavoriteClear1".to_string(),
        iced::keyboard::key::Named::FavoriteClear2 => "FavoriteClear2".to_string(),
        iced::keyboard::key::Named::FavoriteClear3 => "FavoriteClear3".to_string(),
        iced::keyboard::key::Named::FavoriteRecall0 => "FavoriteRecall0".to_string(),
        iced::keyboard::key::Named::FavoriteRecall1 => "FavoriteRecall1".to_string(),
        iced::keyboard::key::Named::FavoriteRecall2 => "FavoriteRecall2".to_string(),
        iced::keyboard::key::Named::FavoriteRecall3 => "FavoriteRecall3".to_string(),
        iced::keyboard::key::Named::FavoriteStore0 => "FavoriteStore0".to_string(),
        iced::keyboard::key::Named::FavoriteStore1 => "FavoriteStore1".to_string(),
        iced::keyboard::key::Named::FavoriteStore2 => "FavoriteStore2".to_string(),
        iced::keyboard::key::Named::FavoriteStore3 => "FavoriteStore3".to_string(),
        iced::keyboard::key::Named::Guide => "Guide".to_string(),
        iced::keyboard::key::Named::GuideNextDay => "GuideNextDay".to_string(),
        iced::keyboard::key::Named::GuidePreviousDay => "GuidePreviousDay".to_string(),
        iced::keyboard::key::Named::Info => "Info".to_string(),
        iced::keyboard::key::Named::InstantReplay => "InstantReplay".to_string(),
        iced::keyboard::key::Named::Link => "Link".to_string(),
        iced::keyboard::key::Named::ListProgram => "ListProgram".to_string(),
        iced::keyboard::key::Named::LiveContent => "LiveContent".to_string(),
        iced::keyboard::key::Named::Lock => "Lock".to_string(),
        iced::keyboard::key::Named::MediaApps => "MediaApps".to_string(),
        iced::keyboard::key::Named::MediaAudioTrack => "MediaAudioTrack".to_string(),
        iced::keyboard::key::Named::MediaLast => "MediaLast".to_string(),
        iced::keyboard::key::Named::MediaSkipBackward => "MediaSkipBackward".to_string(),
        iced::keyboard::key::Named::MediaSkipForward => "MediaSkipForward".to_string(),
        iced::keyboard::key::Named::MediaStepBackward => "MediaStepBackward".to_string(),
        iced::keyboard::key::Named::MediaStepForward => "MediaStepForward".to_string(),
        iced::keyboard::key::Named::MediaTopMenu => "MediaTopMenu".to_string(),
        iced::keyboard::key::Named::NavigateIn => "NavigateIn".to_string(),
        iced::keyboard::key::Named::NavigateNext => "NavigateNext".to_string(),
        iced::keyboard::key::Named::NavigateOut => "NavigateOut".to_string(),
        iced::keyboard::key::Named::NavigatePrevious => "NavigatePrevious".to_string(),
        iced::keyboard::key::Named::NextFavoriteChannel => "NextFavoriteChannel".to_string(),
        iced::keyboard::key::Named::NextUserProfile => "NextUserProfile".to_string(),
        iced::keyboard::key::Named::OnDemand => "OnDemand".to_string(),
        iced::keyboard::key::Named::Pairing => "Pairing".to_string(),
        iced::keyboard::key::Named::PinPDown => "PinPDown".to_string(),
        iced::keyboard::key::Named::PinPMove => "PinPMove".to_string(),
        iced::keyboard::key::Named::PinPToggle => "PinPToggle".to_string(),
        iced::keyboard::key::Named::PinPUp => "PinPUp".to_string(),
        iced::keyboard::key::Named::PlaySpeedDown => "PlaySpeedDown".to_string(),
        iced::keyboard::key::Named::PlaySpeedReset => "PlaySpeedReset".to_string(),
        iced::keyboard::key::Named::PlaySpeedUp => "PlaySpeedUp".to_string(),
        iced::keyboard::key::Named::RandomToggle => "RandomToggle".to_string(),
        iced::keyboard::key::Named::RcLowBattery => "RcLowBattery".to_string(),
        iced::keyboard::key::Named::RecordSpeedNext => "RecordSpeedNext".to_string(),
        iced::keyboard::key::Named::RfBypass => "RfBypass".to_string(),
        iced::keyboard::key::Named::ScanChannelsToggle => "ScanChannelsToggle".to_string(),
        iced::keyboard::key::Named::ScreenModeNext => "ScreenModeNext".to_string(),
        iced::keyboard::key::Named::Settings => "Settings".to_string(),
        iced::keyboard::key::Named::SplitScreenToggle => "SplitScreenToggle".to_string(),
        iced::keyboard::key::Named::STBInput => "STBInput".to_string(),
        iced::keyboard::key::Named::STBPower => "STBPower".to_string(),
        iced::keyboard::key::Named::Subtitle => "Subtitle".to_string(),
        iced::keyboard::key::Named::Teletext => "Teletext".to_string(),
        iced::keyboard::key::Named::VideoModeNext => "VideoModeNext".to_string(),
        iced::keyboard::key::Named::Wink => "Wink".to_string(),
        iced::keyboard::key::Named::ZoomToggle => "ZoomToggle".to_string(),
        iced::keyboard::key::Named::F1 => "F1".to_string(),
        iced::keyboard::key::Named::F2 => "F2".to_string(),
        iced::keyboard::key::Named::F3 => "F3".to_string(),
        iced::keyboard::key::Named::F4 => "F4".to_string(),
        iced::keyboard::key::Named::F5 => "F5".to_string(),
        iced::keyboard::key::Named::F6 => "F6".to_string(),
        iced::keyboard::key::Named::F7 => "F7".to_string(),
        iced::keyboard::key::Named::F8 => "F8".to_string(),
        iced::keyboard::key::Named::F9 => "F9".to_string(),
        iced::keyboard::key::Named::F10 => "F10".to_string(),
        iced::keyboard::key::Named::F11 => "F11".to_string(),
        iced::keyboard::key::Named::F12 => "F12".to_string(),
        iced::keyboard::key::Named::F13 => "F13".to_string(),
        iced::keyboard::key::Named::F14 => "F14".to_string(),
        iced::keyboard::key::Named::F15 => "F15".to_string(),
        iced::keyboard::key::Named::F16 => "F16".to_string(),
        iced::keyboard::key::Named::F17 => "F17".to_string(),
        iced::keyboard::key::Named::F18 => "F18".to_string(),
        iced::keyboard::key::Named::F19 => "F19".to_string(),
        iced::keyboard::key::Named::F20 => "F20".to_string(),
        iced::keyboard::key::Named::F21 => "F21".to_string(),
        iced::keyboard::key::Named::F22 => "F22".to_string(),
        iced::keyboard::key::Named::F23 => "F23".to_string(),
        iced::keyboard::key::Named::F24 => "F24".to_string(),
        iced::keyboard::key::Named::F25 => "F25".to_string(),
        iced::keyboard::key::Named::F26 => "F26".to_string(),
        iced::keyboard::key::Named::F27 => "F27".to_string(),
        iced::keyboard::key::Named::F28 => "F28".to_string(),
        iced::keyboard::key::Named::F29 => "F29".to_string(),
        iced::keyboard::key::Named::F30 => "F30".to_string(),
        iced::keyboard::key::Named::F31 => "F31".to_string(),
        iced::keyboard::key::Named::F32 => "F32".to_string(),
        iced::keyboard::key::Named::F33 => "F33".to_string(),
        iced::keyboard::key::Named::F34 => "F34".to_string(),
        iced::keyboard::key::Named::F35 => "F35".to_string(),
    }
}

pub fn generate_frame(
    image_handle: &image::Handle,
    keys: Vec<String>,
    mouse: Vec<Point>,
) -> image::Handle {
    let current_image = match image_handle {
        image::Handle::Path(id, path_buf) => open(path_buf).expect("Failed to open image"),
        image::Handle::Bytes(id, bytes) => todo!(),
        image::Handle::Rgba {
            id,
            width,
            height,
            pixels,
        } => {
            let mut image = DynamicImage::new_rgba8(*width, *height);

            let mut t = 0;

            for i in 0..*height {
                for j in 0..*width {
                    let pixel = Rgba([pixels[t], pixels[t + 1], pixels[t + 2], pixels[t + 3]]);

                    t += 4;

                    image.put_pixel(j, i, pixel);
                }
            }

            image
        }
    };

    let mouse = mouse
        .iter()
        .map(|p| [(p.x * 2.0) as i32, (p.y * 2.0) as i32]) // 0.0, 0.5, 1.0, 1.5 ... => 0, 1, 2, 3 ...
        .collect();

    let image = model_training::inference::generate(&current_image, keys, mouse);

    image::Handle::from_rgba(WIDTH as u32, HEIGHT as u32, image.into_bytes())
}

#[derive(Default)]
pub struct DataStatus {
    // pub video: bool,
    pub images_from_frames: bool,
    pub resized_images: bool,
    pub hdf5_files: bool,
    pub keys: bool,
}

pub fn check_data(state: &mut State) {
    let data_path = PathBuf::from_str("data").unwrap();

    state.data_status.hdf5_files = check_dir_not_empty(&data_path.join("hdf5_files"));
    state.data_status.keys = check_dir_not_empty(&data_path.join("keys"));

    let images_path = data_path.join("images");

    // state.data_status.video = check_dir_not_empty(&data_path.join("videos"));
    state.data_status.images_from_frames = check_dir_not_empty(&images_path.join("raw"));
    state.data_status.resized_images = check_dir_not_empty(&images_path.join("resized_images"));
}

fn check_dir_not_empty(dir: &PathBuf) -> bool {
    // Что бы было
    fs::create_dir_all(dir).unwrap();

    match fs::read_dir(dir) {
        Ok(entries) =>
        // Итерируемся по записям в директории
        {
            for entry in entries {
                if let Ok(entry) = entry {
                    // Проверяем, является ли запись файлом
                    if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                        return true; // Выходим, если нашли хотя бы один файл
                    }
                }
            }

            false
        }
        Err(_) => false,
    }
}
